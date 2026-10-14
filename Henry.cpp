// henry (fork k-vanio): distributed kangaroo mode for henry_cli (puzzle #135).
// License: GPLv3, see "LICENSE.TXT" file
//
// The binary walks ONE herd with the parameters of the server's HerdAssignment
// and talks to the Go worker only through stdin/stdout. It never resolves a key,
// keeps no DP database and writes no file (CLAUDE.md rules 8 and 17): the walk
// is the v4's, bit for bit as app/internal/kangaroo (the Go reference) models it.
//
//   rckangaroo -henry -gpu G -range R -dp D -dp-extra E -jumps e1,e2,e3 -steps 1000
//              -target <Q' compressed SEC1 hex> -seed <32 B hex> -first-index N
//              -type tame|wild -ckpt-blocks K [-max-blocks M] [-restore]
//   rckangaroo -henry-info -gpu G         HELLO (with KangCnt) and exit 0
//   rckangaroo -henry-selftest <file>     check HenryStartDistance against the
//                                         Go vectors (stderr report, exit 0 = all ok)
//
// PROTOCOL ON STDOUT (contract with app/internal/core/kangproc; binary, little-endian)
//
//   frame: u32 type | u32 payload size | payload
//   1 HELLO       u32 protocol version (1) | u32 KangCnt | u32 STEP_CNT
//   2 DPS         u32 n | n x 56 B: u64 global index | 16 B x (X0 LE, X1 LE) |
//                 24 B d (D[0..2] LE) | u8 type | 3 B zero | u32 restart
//   3 CHECKPOINT  checkpoint v1 blob (kangaroo/checkpoint.go), 32 + 109*KangCnt B:
//                 header  "HKCP" | u16 1 | u8 type | u8 0 | u64 FirstIndex | u64 Count | u64 Blocks
//                 record  u32 Restart | 3 x u64 D | u8 L1S2 | 10 x u64 Ring (oldest first)
//   4 STATS       u64 iterations (Blocks) | u64 jumps since process start | u32 MKeys/s
//   5 FATAL       short text (no secret); the process exits with code != 0
//
// Guarantees (the Go side relies on each of them):
//   1. The first frame is always HELLO. Only a FATAL may come before it (bad
//      arguments, no GPU).
//   2. The DPs of an iteration are written before any CHECKPOINT that includes
//      that iteration. Writes are blocking: a slow reader stalls the GPU instead
//      of losing DPs. A GPU DP buffer overflow (MAX_DP_CNT) is FATAL.
//   3. CHECKPOINT has exactly 32 + 109*KangCnt bytes and Count == KangCnt;
//      Blocks counts iterations since the start of the herd, INCLUDING the ones
//      restored with -restore. One is emitted every K iterations of this process,
//      and a final one on SIGTERM/SIGINT (after the current iteration and its
//      DPs) or when Blocks reaches -max-blocks M; then the process exits 0. The
//      final CHECKPOINT is the last frame.
//      The walk continues from each CHECKPOINT exactly as a resume (-restore)
//      from it would: right after a periodic CHECKPOINT every point is recomputed
//      from its distance (TAME d*G, WILD d*G - Q', negative d = -|d|*G), so a
//      false loop of KernelB (64-bit compare of d) cannot leave a kangaroo off
//      point(d) past a checkpoint, and the server's replay from it matches.
//      Blocks is STRICTLY increasing: a CHECKPOINT is only emitted when Blocks is
//      above the last one emitted or restored in this process. So a stop right
//      after a periodic checkpoint, or -restore of a blob already at Blocks = M
//      with -max-blocks M, emits no new CHECKPOINT: the process sends STATS and
//      exits 0 (the last frame is then STATS or the earlier CHECKPOINT).
//   4. Every emitted DP already passed the dp-extra filter (bits 64..64+E-1 of x,
//      the low bits of X1, are zero) on top of the GPU criterion (bits 224..).
//   5. DPs are emitted exactly as the v4's BuildDP produced them, including the
//      overflow case (more than DPTABLE_MAX_CNT DPs of one kangaroo in one
//      iteration: the v4 pairs the distance with the x of another point). They
//      are neither dropped nor corrected here; the server refuses them by the
//      scalar multiplication.
//
// STDIN: with -restore, the checkpoint v1 blob and nothing else: the process
// reads it, then requires EOF (the worker closes stdin). Refused (FATAL): wrong
// magic/version/type/first index/count, Blocks >= 2^64 - 2^32, any Restart != 0,
// L1S2 not 0/1, |d| >= 2^(R+16), TAME d = 0, trailing bytes.
// Human-readable log: stderr only. stdout (fd 1) is redirected to stderr at
// startup and the protocol uses a private duplicate of the original fd 1, so no
// printf left in the v4 code can corrupt the frame stream.

#include <cstddef>
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "cuda_runtime.h"

#include "defs.h"
#include "utils.h"
#include "Ec.h"
#include "GpuKang.h"
#include "Henry.h"
#include "HenryStart.h"

extern EcJMP EcJumps1[JMP_CNT];
extern EcJMP EcJumps2[JMP_CNT];
extern EcJMP EcJumps3[JMP_CNT];
void GenerateJumpTables(int e1, int e2, int e3);

enum
{
	FRAME_HELLO = 1,
	FRAME_DPS = 2,
	FRAME_CHECKPOINT = 3,
	FRAME_STATS = 4,
	FRAME_FATAL = 5,
};

#define HENRY_PROTOCOL_VERSION	1
#define HENRY_DP_REC_SIZE		56
#define CKPT_HEADER_SIZE		32
#define CKPT_RECORD_SIZE		(4 + 24 + 1 + 8 * MD_LEN)
#define CKPT_VERSION			1
// |d| bound of a checkpoint record: 2^(R + 16), as CheckState in checkpoint.go
#define DIST_SLACK_BITS			16
#define STATS_INTERVAL_MS		10000

// The turbo kernels (kernel_sm89/sm120.cubin) read TKparams by fixed offsets
// (main.asm/newKernelB.asm: c[0x0][0x160 + offset]); the henry fields were only
// appended. A change above them would silently break the turbo walk.
static_assert(offsetof(TKparams, L2) == 0x00, "TKparams layout (L2)");
static_assert(offsetof(TKparams, L1S2) == 0x38, "TKparams layout (L1S2)");
static_assert(offsetof(TKparams, iter_cnt) == 0x48, "TKparams layout (iter_cnt)");
static_assert(offsetof(TKparams, dp_mask) == 0x54, "TKparams layout (dp_mask)");
static_assert(offsetof(TKparams, LoopTable) == 0x60, "TKparams layout (LoopTable)");
static_assert(offsetof(TKparams, dists) == 0x70, "TKparams layout (dists)");
static_assert(offsetof(TKparams, KangCnt) == 0x80, "TKparams layout (KangCnt)");
static_assert(offsetof(TKparams, HenryMode) > offsetof(TKparams, KernelC_LDS_Size), "henry fields must stay at the end");

static int gProtoFd = -1;
static volatile sig_atomic_t gStopRequested = 0;

static void OnStopSignal(int)
{
	gStopRequested = 1;
}

//////////////////////////////////////////////////////////////////////////////
// output

static bool WriteAll(int fd, const void* data, size_t len)
{
	const u8* p = (const u8*)data;
	while (len)
	{
		ssize_t n = write(fd, p, len);
		if (n < 0)
		{
			if (errno == EINTR)
				continue;
			if (errno == EAGAIN || errno == EWOULDBLOCK)
			{
				// stdout handed over in non-blocking mode: wait instead of failing
				struct pollfd pfd = { fd, POLLOUT, 0 };
				if (poll(&pfd, 1, -1) >= 0 || errno == EINTR)
					continue;
			}
			return false;
		}
		p += n;
		len -= (size_t)n;
	}
	return true;
}

// A write that fails midway leaves a partial frame on the stream: from then on
// nothing more may go to the protocol fd (not even FATAL), or the reader would
// decode garbage. Fatal() then only logs to stderr.
static bool SendFrame(u32 type, const void* payload, u32 size)
{
	if (gProtoFd < 0)
		return false;
	u32 hdr[2] = { type, size };
	if (WriteAll(gProtoFd, hdr, sizeof(hdr)) && (!size || WriteAll(gProtoFd, payload, size)))
		return true;
	gProtoFd = -1;
	return false;
}

// Fatal logs to stderr, sends a FATAL frame when the protocol is up and returns
// the exit code. The text must never carry secrets (rule 8); there are none in
// this process anyway: no key is ever found or handled here.
static int Fatal(const char* fmt, ...)
{
	char msg[512];
	va_list ap;
	va_start(ap, fmt);
	vsnprintf(msg, sizeof(msg), fmt, ap);
	va_end(ap);
	fprintf(stderr, "henry: FATAL: %s\n", msg);
	if (gProtoFd >= 0)
		SendFrame(FRAME_FATAL, msg, (u32)strlen(msg));
	return 2;
}

static void PutU16(u8* p, u16 v) { p[0] = (u8)v; p[1] = (u8)(v >> 8); }
static void PutU32(u8* p, u32 v) { for (int i = 0; i < 4; i++) p[i] = (u8)(v >> (8 * i)); }
static void PutU64(u8* p, u64 v) { for (int i = 0; i < 8; i++) p[i] = (u8)(v >> (8 * i)); }
static u16 GetU16(const u8* p) { return (u16)(p[0] | (p[1] << 8)); }
static u32 GetU32(const u8* p) { u32 v = 0; for (int i = 3; i >= 0; i--) v = (v << 8) | p[i]; return v; }
static u64 GetU64(const u8* p) { u64 v = 0; for (int i = 7; i >= 0; i--) v = (v << 8) | p[i]; return v; }

//////////////////////////////////////////////////////////////////////////////
// argument parsing (strict: anything malformed is FATAL, never a default)

static bool ParseU64Dec(const char* s, u64 min, u64 max, u64* out)
{
	if (!s || !*s || strlen(s) > 20)
		return false;
	u64 v = 0;
	for (const char* p = s; *p; p++)
	{
		if (*p < '0' || *p > '9')
			return false;
		u64 d = (u64)(*p - '0');
		if (v > (0xFFFFFFFFFFFFFFFFull - d) / 10)
			return false;
		v = v * 10 + d;
	}
	if (v < min || v > max)
		return false;
	*out = v;
	return true;
}

static bool ParseIntDec(const char* s, int min, int max, int* out)
{
	u64 v;
	if (!ParseU64Dec(s, (u64)min, (u64)max, &v))
		return false;
	*out = (int)v;
	return true;
}

static int HexVal(char c)
{
	if (c >= '0' && c <= '9') return c - '0';
	if (c >= 'a' && c <= 'f') return c - 'a' + 10;
	if (c >= 'A' && c <= 'F') return c - 'A' + 10;
	return -1;
}

// exactly 2*n hex digits into n bytes (big-endian text order)
static bool ParseHexBytes(const char* s, u8* out, int n)
{
	if (!s || (int)strlen(s) != 2 * n)
		return false;
	for (int i = 0; i < n; i++)
	{
		int h = HexVal(s[2 * i]), l = HexVal(s[2 * i + 1]);
		if (h < 0 || l < 0)
			return false;
		out[i] = (u8)((h << 4) | l);
	}
	return true;
}

struct HenryArgs
{
	bool info = false;
	int gpu = 0;
	int range = 0;
	int dp = -1;
	int dp_extra = -1;
	int jumps[3] = { 0, 0, 0 };
	int steps = 0;
	bool has_target = false;
	EcPoint target; // Q' (already prepared by the server)
	bool has_seed = false;
	u8 seed[32];
	bool has_first = false;
	u64 first_index = 0;
	int type = -1;
	u64 ckpt_blocks = 0;
	u64 max_blocks = 0; // 0 = no limit
	bool restore = false;
};

static bool ParseJumps(const char* s, int e[3])
{
	if (!s)
		return false;
	char buf[64];
	if (strlen(s) >= sizeof(buf))
		return false;
	strcpy(buf, s);
	char* parts[3];
	int n = 0;
	char* p = buf;
	for (;;)
	{
		if (n == 3)
			return false;
		parts[n++] = p;
		char* c = strchr(p, ',');
		if (!c)
			break;
		*c = 0;
		p = c + 1;
	}
	if (n != 3)
		return false;
	// jumps.go JumpSpecFromExps: each exponent in [1, 180]
	for (int i = 0; i < 3; i++)
		if (!ParseIntDec(parts[i], 1, 180, &e[i]))
			return false;
	return true;
}

static bool ParseTarget(const char* s, EcPoint* out)
{
	if (!s || strlen(s) != 66)
		return false;
	char up[67];
	for (int i = 0; i < 66; i++)
	{
		if (HexVal(s[i]) < 0)
			return false;
		up[i] = (s[i] >= 'a' && s[i] <= 'f') ? (char)(s[i] - 'a' + 'A') : s[i];
	}
	up[66] = 0;
	if (up[0] != '0' || (up[1] != '2' && up[1] != '3'))
		return false;
	// x must be a field element: EcPoint::SetHexStr does not check x < p
	EcInt x, p;
	if (!x.SetHexStr(up + 2))
		return false;
	p.SetHexStr("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F");
	if (!x.IsLessThanU(p))
		return false;
	EcPoint q;
	if (!q.SetHexStr(up)) // decompresses and checks the curve equation
		return false;
	*out = q;
	return true;
}

// returns NULL on success, else the error text
static const char* ParseHenryArgs(int argc, char* argv[], HenryArgs* a)
{
	static char err[256];
	bool seen_gpu = false, seen_range = false, seen_jumps = false, seen_ckpt = false, seen_max = false, seen_mode = false;
	for (int i = 1; i < argc; i++)
	{
		const char* opt = argv[i];
		const char* val = (i + 1 < argc) ? argv[i + 1] : NULL;
		bool ok = true, takes_val = true, dup = false;
		if (!strcmp(opt, "-henry") || !strcmp(opt, "-henry-info"))
		{
			takes_val = false;
			dup = seen_mode;
			seen_mode = true;
			a->info = !strcmp(opt, "-henry-info");
		}
		else if (!strcmp(opt, "-restore"))
		{
			takes_val = false;
			dup = a->restore;
			a->restore = true;
		}
		else if (!strcmp(opt, "-gpu")) { dup = seen_gpu; seen_gpu = true; ok = ParseIntDec(val, 0, MAX_GPU_CNT - 1, &a->gpu); }
		else if (!strcmp(opt, "-range")) { dup = seen_range; seen_range = true; ok = ParseIntDec(val, 16, 170, &a->range); }
		else if (!strcmp(opt, "-dp")) { dup = a->dp >= 0; ok = ParseIntDec(val, 1, 32, &a->dp); }
		else if (!strcmp(opt, "-dp-extra")) { dup = a->dp_extra >= 0; ok = ParseIntDec(val, 0, 32, &a->dp_extra); }
		else if (!strcmp(opt, "-jumps")) { dup = seen_jumps; seen_jumps = true; ok = ParseJumps(val, a->jumps); }
		else if (!strcmp(opt, "-steps")) { dup = a->steps != 0; ok = ParseIntDec(val, 1, 1000000000, &a->steps); }
		else if (!strcmp(opt, "-target")) { dup = a->has_target; a->has_target = true; ok = ParseTarget(val, &a->target); }
		else if (!strcmp(opt, "-seed")) { dup = a->has_seed; a->has_seed = true; ok = ParseHexBytes(val, a->seed, 32); }
		else if (!strcmp(opt, "-first-index")) { dup = a->has_first; a->has_first = true; ok = ParseU64Dec(val, 0, 0xFFFFFFFFFFFFFFFFull, &a->first_index); }
		else if (!strcmp(opt, "-type"))
		{
			dup = a->type >= 0;
			if (val && !strcmp(val, "tame")) a->type = TAME;
			else if (val && !strcmp(val, "wild")) a->type = WILD;
			else ok = false;
		}
		else if (!strcmp(opt, "-ckpt-blocks")) { dup = seen_ckpt; seen_ckpt = true; ok = ParseU64Dec(val, 1, 1000000000ull, &a->ckpt_blocks); }
		else if (!strcmp(opt, "-max-blocks")) { dup = seen_max; seen_max = true; ok = ParseU64Dec(val, 1, 1ull << 62, &a->max_blocks); }
		else
		{
			snprintf(err, sizeof(err), "unknown or unsupported option in -henry mode: %.64s", opt);
			return err;
		}
		if (dup)
		{
			snprintf(err, sizeof(err), "option given twice: %s", opt);
			return err;
		}
		if (!ok)
		{
			snprintf(err, sizeof(err), "invalid or missing value for %s", opt);
			return err;
		}
		if (takes_val)
			i++;
	}
	if (a->info)
	{
		if (seen_range || a->dp >= 0 || a->dp_extra >= 0 || seen_jumps || a->steps || a->has_target || a->has_seed ||
			a->has_first || a->type >= 0 || seen_ckpt || seen_max || a->restore)
			return "-henry-info takes only -gpu";
		return NULL;
	}
	if (!seen_range || a->dp < 0 || a->dp_extra < 0 || !seen_jumps || !a->steps || !a->has_target || !a->has_seed ||
		!a->has_first || a->type < 0 || !seen_ckpt)
		return "-henry needs -range -dp -dp-extra -jumps -steps -target -seed -first-index -type -ckpt-blocks";
	// The walk depends on STEP_CNT (when loops >2 escape), so a job built for
	// another value would produce DPs nobody else reproduces (rule 15).
	if (a->steps != STEP_CNT)
	{
		snprintf(err, sizeof(err), "-steps %d differs from this binary's STEP_CNT %d", a->steps, STEP_CNT);
		return err;
	}
	return NULL;
}

//////////////////////////////////////////////////////////////////////////////
// GPU layout <-> checkpoint record (per kangaroo, kang_ind = THREAD_X +
// BLOCK_X*BLOCK_SIZE + group*BLOCK_SIZE*BlockCnt, the index KernelGen/KernelA/
// KernelC/BuildDP use)
//
// L1S2: one u32 per thread (BLOCK_X*BLOCK_SIZE + THREAD_X), bit = group
//   (RCGpuCore.cu:61/131-141/168; main.asm: LDG/STG [gID*4 + L1S2], bMask = 1 << group).
// LoopTable: KernelB walks two kangaroos per thread, A = even and B = odd
//   source thread of group gr_ind = tind/128, tind = THREAD_X + gr_ind2*BLOCK_SIZE
//   (RCGpuCore.cu:294-314, newKernelB.asm "calc original kang_ind"), and keeps the
//   last MD_LEN low limbs of d at
//   MD_LEN*BLOCK_SIZE*PNT_GROUP_CNT*BLOCK_X + 2*MD_LEN*BLOCK_SIZE*gr_ind2 + (slot + MD_LEN*isB)*BLOCK_SIZE + THREAD_X
//   (RCGpuCore.cu:299-300/354-356, newKernelB.asm SaveRegAB and "load regs").
//   At the end of KernelB the window is rotated so that slot 0 is the OLDEST
//   entry ("store so cur_ind is 0 at next loading"), which is the checkpoint's
//   Ring order; a kangaroo that looped mid-iteration stops updating its window
//   and the rotation uses its own cur_ind, so the order still holds.

struct GpuMap
{
	u32 block_cnt;
	u32 kang_cnt;

	u32 L1S2Word(u32 k) const { return k % (block_cnt * BLOCK_SIZE); }
	u32 L1S2Bit(u32 k) const { return k / (block_cnt * BLOCK_SIZE); }
	u64 LoopIndex(u32 k, int slot) const
	{
		u32 g = k / (block_cnt * BLOCK_SIZE);
		u32 r = k % (block_cnt * BLOCK_SIZE);
		u32 bx = r / BLOCK_SIZE;
		u32 t = r % BLOCK_SIZE;
		u32 is_b = t & 1;
		u32 tind = g * (BLOCK_SIZE / 2) + (t >> 1);
		u32 thr = tind % BLOCK_SIZE;
		u32 gr2 = tind / BLOCK_SIZE;
		return (u64)MD_LEN * BLOCK_SIZE * PNT_GROUP_CNT * bx + 2ull * MD_LEN * BLOCK_SIZE * gr2 +
			(u64)(slot + MD_LEN * is_b) * BLOCK_SIZE + thr;
	}
};

// The mapping must be a bijection onto the LoopTable; checked once at start so a
// change of BLOCK_SIZE/PNT_GROUP_CNT cannot silently scramble checkpoints.
static bool CheckLoopMapping(const GpuMap& m)
{
	u64 n = (u64)m.kang_cnt * MD_LEN;
	std::vector<u8> seen(n, 0);
	for (u32 k = 0; k < m.kang_cnt; k++)
		for (int j = 0; j < MD_LEN; j++)
		{
			u64 idx = m.LoopIndex(k, j);
			if (idx >= n || seen[idx])
				return false;
			seen[idx] = 1;
		}
	return true;
}

//////////////////////////////////////////////////////////////////////////////
// distances

static bool DistIsNeg(const u64* d) { return (d[2] >> 63) != 0; }

static void DistAbs(const u64* d, u64* out)
{
	if (!DistIsNeg(d))
	{
		out[0] = d[0]; out[1] = d[1]; out[2] = d[2];
		return;
	}
	u8 c = _subborrow_u64(0, 0, d[0], (unsigned long long*)&out[0]);
	c = _subborrow_u64(c, 0, d[1], (unsigned long long*)&out[1]);
	_subborrow_u64(c, 0, d[2], (unsigned long long*)&out[2]);
}

static int BitLen192(const u64* v)
{
	for (int i = 2; i >= 0; i--)
		if (v[i])
			return 64 * i + 64 - __builtin_clzll(v[i]);
	return 0;
}

// Point of a kangaroo at signed distance d: TAME d*G, WILD d*G - Q' (d = 0: -Q').
// CPU reference for the start-point self-check of KernelGen.
static EcPoint PointAt(const u64* d, int type, EcPoint& pnt_wild)
{
	u64 mag[3];
	DistAbs(d, mag);
	if (!(mag[0] | mag[1] | mag[2]))
		return pnt_wild; // only reachable for WILD (TAME d = 0 is refused before)
	EcInt k;
	k.SetZero();
	k.data[0] = mag[0]; k.data[1] = mag[1]; k.data[2] = mag[2];
	EcPoint p = Ec::MultiplyG(k);
	if (DistIsNeg(d))
		p.y.NegModP();
	if (type == WILD)
		p = Ec::AddPoints(p, pnt_wild);
	return p;
}

//////////////////////////////////////////////////////////////////////////////
// checkpoint

struct HerdState
{
	std::vector<u64> dists3;     // KangCnt*3, signed 192-bit
	std::vector<u32> restarts;   // KangCnt
	std::vector<u32> l1s2;       // BlockCnt*BLOCK_SIZE (GPU layout)
	std::vector<u64> loop_table; // KangCnt*MD_LEN (GPU layout)
	u64 blocks = 0;
};

static bool ReadExact(int fd, u8* buf, size_t len)
{
	while (len)
	{
		ssize_t n = read(fd, buf, len);
		if (n < 0)
		{
			if (errno == EINTR)
				continue;
			return false;
		}
		if (n == 0)
			return false;
		buf += n;
		len -= (size_t)n;
	}
	return true;
}

static const char* LoadCheckpoint(const HenryArgs& a, const GpuMap& m, HerdState* st)
{
	static char err[256];
	u64 size = CKPT_HEADER_SIZE + (u64)CKPT_RECORD_SIZE * m.kang_cnt;
	std::vector<u8> blob(size);
	if (!ReadExact(0, blob.data(), size))
	{
		snprintf(err, sizeof(err), "checkpoint on stdin is truncated (need %llu bytes)", (unsigned long long)size);
		return err;
	}
	const u8* h = blob.data();
	if (memcmp(h, "HKCP", 4) || GetU16(h + 4) != CKPT_VERSION)
		return "checkpoint: bad magic or version";
	if (h[6] != (u8)a.type || h[7] != 0)
		return "checkpoint: kangaroo type differs from -type";
	if (GetU64(h + 8) != a.first_index)
		return "checkpoint: first index differs from -first-index";
	if (GetU64(h + 16) != m.kang_cnt)
		return "checkpoint: kangaroo count differs from this GPU's KangCnt";
	st->blocks = GetU64(h + 24);
	// Far from any real count; refusing it keeps st.blocks++ from wrapping.
	if (st->blocks >= 0xFFFFFFFF00000000ull)
		return "checkpoint: Blocks out of range";
	// stdin carries the blob and nothing else: trailing bytes mean a framing bug
	// on the worker side, and silently ignoring them could resume the wrong state.
	u8 extra;
	ssize_t got;
	do
		got = read(0, &extra, 1);
	while (got < 0 && errno == EINTR);
	if (got != 0)
		return got > 0 ? "checkpoint: extra bytes on stdin after the blob" : "checkpoint: read error on stdin";
	for (u32 k = 0; k < m.kang_cnt; k++)
	{
		const u8* r = h + CKPT_HEADER_SIZE + (u64)CKPT_RECORD_SIZE * k;
		u64* d = &st->dists3[3 * (u64)k];
		st->restarts[k] = GetU32(r);
		// F3 has no kangaroo restart (the server does not issue one yet): a record
		// with Restart != 0 cannot come from this herd's walk.
		if (st->restarts[k] != 0)
		{
			snprintf(err, sizeof(err), "checkpoint: kangaroo %u has Restart = %u (restarts are not supported)", k, st->restarts[k]);
			return err;
		}
		d[0] = GetU64(r + 4); d[1] = GetU64(r + 12); d[2] = GetU64(r + 20);
		u8 l1s2 = r[28];
		if (l1s2 > 1)
		{
			snprintf(err, sizeof(err), "checkpoint: kangaroo %u has L1S2 = %u", k, l1s2);
			return err;
		}
		u64 mag[3];
		DistAbs(d, mag);
		if (BitLen192(mag) > a.range + DIST_SLACK_BITS) // |d| < 2^(R+16), as the Go side
		{
			snprintf(err, sizeof(err), "checkpoint: kangaroo %u distance out of bounds", k);
			return err;
		}
		if (a.type == TAME && !(mag[0] | mag[1] | mag[2]))
		{
			snprintf(err, sizeof(err), "checkpoint: TAME kangaroo %u at distance 0 (point at infinity)", k);
			return err;
		}
		if (l1s2)
			st->l1s2[m.L1S2Word(k)] |= 1u << m.L1S2Bit(k);
		for (int j = 0; j < MD_LEN; j++)
			st->loop_table[m.LoopIndex(k, j)] = GetU64(r + 29 + 8 * j);
	}
	return NULL;
}

// Reads the GPU state at an iteration boundary and sends it as CHECKPOINT v1.
static const char* SendCheckpoint(RCGpuKang* kang, const HenryArgs& a, const GpuMap& m, const HerdState& st,
	std::vector<u64>& dists4, std::vector<u32>& l1s2, std::vector<u64>& lt, std::vector<u8>& blob)
{
	static char err[256];
	if (!kang->HenryReadState(dists4.data(), l1s2.data(), lt.data(), err, sizeof(err)))
		return err;
	u8* h = blob.data();
	memcpy(h, "HKCP", 4);
	PutU16(h + 4, CKPT_VERSION);
	h[6] = (u8)a.type;
	h[7] = 0;
	PutU64(h + 8, a.first_index);
	PutU64(h + 16, m.kang_cnt);
	PutU64(h + 24, st.blocks);
	for (u32 k = 0; k < m.kang_cnt; k++)
	{
		u8* r = h + CKPT_HEADER_SIZE + (u64)CKPT_RECORD_SIZE * k;
		PutU32(r, st.restarts[k]);
		PutU64(r + 4, dists4[4 * (u64)k + 0]);
		PutU64(r + 12, dists4[4 * (u64)k + 1]);
		PutU64(r + 20, dists4[4 * (u64)k + 2]);
		r[28] = (u8)((l1s2[m.L1S2Word(k)] >> m.L1S2Bit(k)) & 1);
		for (int j = 0; j < MD_LEN; j++)
			PutU64(r + 29 + 8 * j, lt[m.LoopIndex(k, j)]);
	}
	if (!SendFrame(FRAME_CHECKPOINT, blob.data(), (u32)blob.size()))
		return "write to stdout failed";
	return NULL;
}

static bool SendStats(u64 blocks, u64 jumps, u32 mkeys)
{
	u8 p[20];
	PutU64(p, blocks);
	PutU64(p + 8, jumps);
	PutU32(p + 16, mkeys);
	return SendFrame(FRAME_STATS, p, sizeof(p));
}

//////////////////////////////////////////////////////////////////////////////
// self-test of HenryStartDistance against the vectors generated by the Go code

static int HenrySelfTest(const char* path)
{
	FILE* f = fopen(path, "r"); // read only; the -henry mode itself opens no file
	if (!f)
	{
		fprintf(stderr, "selftest: cannot open %s\n", path);
		return 1;
	}
	char line[512];
	int total = 0, bad = 0;
	while (fgets(line, sizeof(line), f))
	{
		if (line[0] == '#' || line[0] == '\n' || line[0] == '\r')
			continue;
		char seed_hex[100], type[16], d_hex[100];
		int range;
		unsigned long long index;
		unsigned int restart;
		if (sscanf(line, "%d %99s %15s %llu %u %99s", &range, seed_hex, type, &index, &restart, d_hex) != 6)
		{
			fprintf(stderr, "selftest: malformed line: %s", line);
			bad++;
			continue;
		}
		u8 seed[32];
		if (!ParseHexBytes(seed_hex, seed, 32) || range < 6 || range > 170 || (strcmp(type, "tame") && strcmp(type, "wild")))
		{
			fprintf(stderr, "selftest: malformed line: %s", line);
			bad++;
			continue;
		}
		u64 d[3];
		HenryStartDistance(seed, range, strcmp(type, "tame") ? WILD : TAME, index, restart, d);
		char got[64];
		if (d[2])
			snprintf(got, sizeof(got), "%llx%016llx%016llx", (unsigned long long)d[2], (unsigned long long)d[1], (unsigned long long)d[0]);
		else if (d[1])
			snprintf(got, sizeof(got), "%llx%016llx", (unsigned long long)d[1], (unsigned long long)d[0]);
		else
			snprintf(got, sizeof(got), "%llx", (unsigned long long)d[0]);
		for (char* p = d_hex; *p; p++)
			if (*p >= 'A' && *p <= 'F')
				*p = (char)(*p - 'A' + 'a');
		total++;
		if (strcmp(got, d_hex))
		{
			bad++;
			fprintf(stderr, "selftest: MISMATCH range %d %s index %llu restart %u: want %s got %s\n", range, type, index, restart, d_hex, got);
		}
	}
	fclose(f);
	fprintf(stderr, "selftest: %d vectors, %d mismatches\n", total, bad);
	return (total > 0 && bad == 0) ? 0 : 1;
}

//////////////////////////////////////////////////////////////////////////////

bool IsHenryInvocation(int argc, char* argv[])
{
	for (int i = 1; i < argc; i++)
		if (!strcmp(argv[i], "-henry") || !strcmp(argv[i], "-henry-info") || !strcmp(argv[i], "-henry-selftest"))
			return true;
	return false;
}

static RCGpuKang* OpenGpu(int idx, char* err, int err_size)
{
	int gcnt = 0;
	cudaError_t e = cudaGetDeviceCount(&gcnt);
	if (e != cudaSuccess || gcnt <= 0)
	{
		snprintf(err, err_size, "no CUDA device (%s)", e != cudaSuccess ? cudaGetErrorString(e) : "count 0");
		return NULL;
	}
	if (idx >= gcnt)
	{
		snprintf(err, err_size, "-gpu %d but only %d CUDA devices", idx, gcnt);
		return NULL;
	}
	e = cudaSetDevice(idx);
	if (e != cudaSuccess)
	{
		snprintf(err, err_size, "cudaSetDevice(%d): %s", idx, cudaGetErrorString(e));
		return NULL;
	}
	cudaDeviceProp prop;
	e = cudaGetDeviceProperties(&prop, idx);
	if (e != cudaSuccess)
	{
		snprintf(err, err_size, "cudaGetDeviceProperties(%d): %s", idx, cudaGetErrorString(e));
		return NULL;
	}
	if (prop.major < 6)
	{
		snprintf(err, err_size, "GPU %d (cap %d.%d) not supported", idx, prop.major, prop.minor);
		return NULL;
	}
	cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
	// same choices as InitGpus (RCKangaroo.cpp)
	RCGpuKang* k = new RCGpuKang();
	k->CudaIndex = idx;
	k->persistingL2CacheMaxSize = prop.persistingL2CacheMaxSize;
	k->mpCnt = prop.multiProcessorCount;
	k->JumperInd = 0;
	int cm = prop.major * 10 + prop.minor;
	k->Is5xxx = false;
	if (cm != 89 && cm != 120)
		k->sm_inv_cnt = 0;
	else
	{
		k->Is5xxx = (prop.major == 12);
		k->sm_inv_cnt = k->Is5xxx ? (k->mpCnt / 24) : (k->mpCnt / 32);
		if (!k->sm_inv_cnt)
			k->sm_inv_cnt = 1;
	}
	fprintf(stderr, "henry: GPU %d: %s, %d SMs, cap %d.%d, %s kernels\n", idx, prop.name, prop.multiProcessorCount,
		prop.major, prop.minor, k->sm_inv_cnt ? "turbo" : "CUDA");
	return k;
}

static int RunHerd(const HenryArgs& a, RCGpuKang* kang, u32 kang_cnt);

int HenryMain(int argc, char* argv[])
{
	for (int i = 1; i < argc; i++)
		if (!strcmp(argv[i], "-henry-selftest"))
		{
			if (argc != 3 || i != 1)
			{
				fprintf(stderr, "usage: rckangaroo -henry-selftest <vectors file>\n");
				return 1;
			}
			return HenrySelfTest(argv[2]);
		}

	// Keep a private handle to the real stdout for the frames and point fd 1 at
	// stderr: every printf of the v4 code (Prepare, CallCubin, ...) then lands in
	// the log instead of in the middle of a frame.
	gProtoFd = fcntl(1, F_DUPFD_CLOEXEC, 3);
	if (gProtoFd < 0 || dup2(2, 1) < 0)
	{
		fprintf(stderr, "henry: FATAL: cannot set up stdout\n");
		return 2;
	}
	setvbuf(stdout, NULL, _IOLBF, 0); // the v4 log lines interleave in order with ours
	signal(SIGPIPE, SIG_IGN); // a closed pipe becomes a write error, not a silent kill
	struct sigaction sa;
	memset(&sa, 0, sizeof(sa));
	sa.sa_handler = OnStopSignal;
	sigemptyset(&sa.sa_mask);
	sa.sa_flags = SA_RESTART;
	sigaction(SIGTERM, &sa, NULL);
	sigaction(SIGINT, &sa, NULL);

	InitEc();
	HenryArgs a;
	const char* perr = ParseHenryArgs(argc, argv, &a);
	if (perr)
		return Fatal("%s", perr);

	char err[256];
	RCGpuKang* kang = OpenGpu(a.gpu, err, sizeof(err));
	if (!kang)
		return Fatal("%s", err);
	u32 kang_cnt = BLOCK_SIZE * PNT_GROUP_CNT * (u32)(kang->mpCnt - kang->sm_inv_cnt);
	u32 hello[3] = { HENRY_PROTOCOL_VERSION, kang_cnt, STEP_CNT };
	if (!SendFrame(FRAME_HELLO, hello, sizeof(hello)))
		return Fatal("write to stdout failed");
	if (a.info)
		return 0;
	// last global index = first + KangCnt - 1 must fit in 64 bits
	if (a.first_index > 0xFFFFFFFFFFFFFFFFull - (kang_cnt - 1))
		return Fatal("-first-index + KangCnt - 1 overflows 64 bits");
	return RunHerd(a, kang, kang_cnt);
}

static int RunHerd(const HenryArgs& a, RCGpuKang* kang, u32 kang_cnt)
{
	char err[256];
	GenerateJumpTables(a.jumps[0], a.jumps[1], a.jumps[2]);

	EcPoint pnt_wild = a.target; // -Q' (GpuKang.cpp Start: PntWild)
	pnt_wild.y.NegModP();

	kang->HenryMode = true;
	kang->HenryWild = (a.type == WILD);
	if (!kang->Prepare(a.target, a.range, a.dp, EcJumps1, EcJumps2, EcJumps3))
		return Fatal("GPU %d prepare failed (see log)", a.gpu);
	if ((u32)kang->KangCnt != kang_cnt)
		return Fatal("KangCnt mismatch (%d vs %u)", kang->KangCnt, kang_cnt);

	GpuMap m;
	m.block_cnt = (u32)kang->GetBlockCnt();
	m.kang_cnt = kang_cnt;
	if (!CheckLoopMapping(m))
		return Fatal("LoopTable mapping is not a bijection: GPU layout constants changed");

	HerdState st;
	st.dists3.assign(3 * (u64)kang_cnt, 0);
	st.restarts.assign(kang_cnt, 0);
	st.l1s2.assign((u64)m.block_cnt * BLOCK_SIZE, 0);
	st.loop_table.assign((u64)kang_cnt * MD_LEN, 0); // the v4 starts with a zeroed LoopTable
	if (a.restore)
	{
		const char* lerr = LoadCheckpoint(a, m, &st);
		if (lerr)
			return Fatal("%s", lerr);
		if (a.max_blocks && st.blocks > a.max_blocks)
			return Fatal("checkpoint is at %llu iterations, beyond -max-blocks %llu", (unsigned long long)st.blocks, (unsigned long long)a.max_blocks);
		fprintf(stderr, "henry: restored %u kangaroos at %llu iterations\n", kang_cnt, (unsigned long long)st.blocks);
	}
	else
	{
		for (u32 k = 0; k < kang_cnt; k++)
			HenryStartDistance(a.seed, a.range, a.type, a.first_index + k, 0, &st.dists3[3 * (u64)k]);
	}

	if (!kang->HenryStart(st.dists3.data(), st.l1s2.data(), st.loop_table.data(), pnt_wild, err, sizeof(err)))
		return Fatal("%s", err);

	// Self-check of KernelGen against the CPU on a few kangaroos, including the
	// negative-distance and WILD d = 0 paths when present: a wrong start point
	// would make every DP of the herd worthless.
	{
		std::vector<u32> sample = { 0, kang_cnt - 1, kang_cnt / 2, 12345 % kang_cnt, 777777 % kang_cnt };
		for (u32 k = 0; k < kang_cnt; k++)
			if (DistIsNeg(&st.dists3[3 * (u64)k])) { sample.push_back(k); break; }
		for (u32 k = 0; k < kang_cnt; k++)
		{
			const u64* d = &st.dists3[3 * (u64)k];
			if (!(d[0] | d[1] | d[2])) { sample.push_back(k); break; }
		}
		for (u32 k : sample)
		{
			EcPoint gpu;
			if (!kang->HenryReadPoint((int)k, &gpu))
				return Fatal("cannot read start point back from the GPU");
			EcPoint cpu = PointAt(&st.dists3[3 * (u64)k], a.type, pnt_wild);
			if (!cpu.IsEqual(gpu))
				return Fatal("start point of kangaroo %u differs between GPU and CPU", k);
		}
	}

	std::vector<u64> dists4(4 * (u64)kang_cnt);
	std::vector<u32> l1s2_buf((u64)m.block_cnt * BLOCK_SIZE);
	std::vector<u64> lt_buf((u64)kang_cnt * MD_LEN);
	std::vector<u8> blob(CKPT_HEADER_SIZE + (u64)CKPT_RECORD_SIZE * kang_cnt);
	std::vector<u8> dps_frame(4 + (u64)HENRY_DP_REC_SIZE * MAX_DP_CNT);
	u64 extra_mask = a.dp_extra ? ((1ull << a.dp_extra) - 1) : 0;
	u64 jumps_per_iter = (u64)kang_cnt * STEP_CNT;

	u64 iters = 0, since_ckpt = 0;
	// The server requires Blocks strictly increasing across checkpoints: a
	// CHECKPOINT only goes out when Blocks is above the last one emitted or
	// restored by this process.
	bool have_ckpt = a.restore;
	u64 last_ckpt_blocks = st.blocks;
	u64 tm_stats = GetTickCount64(), iters_at_stats = 0;
	u32 last_speed = 0;
	fprintf(stderr, "henry: herd %s, first index %llu, %u kangaroos, range %d, dp %d+%d, jumps 2^%d/2^%d/2^%d, checkpoint every %llu\n",
		a.type == TAME ? "TAME" : "WILD", (unsigned long long)a.first_index, kang_cnt, a.range, a.dp, a.dp_extra,
		a.jumps[0], a.jumps[1], a.jumps[2], (unsigned long long)a.ckpt_blocks);

	for (;;)
	{
		bool done = gStopRequested || (a.max_blocks && st.blocks >= a.max_blocks);
		if (!done)
		{
			u32 cnt = 0;
			u32* dps = NULL;
			if (!kang->HenryIterate(&cnt, &dps, err, sizeof(err)))
				return Fatal("%s", err);
			st.blocks++;
			iters++;
			since_ckpt++;

			u32 n = 0;
			for (u32 i = 0; i < cnt; i++)
			{
				const u8* p = (const u8*)dps + (u64)GPU_DP_SIZE * i;
				u32 kind = GetU32(p + 40);
				if (kind >= kang_cnt)
					return Fatal("GPU returned a DP with kangaroo index %u >= %u", kind, kang_cnt);
				if (GetU64(p + 8) & extra_mask)
					continue; // dp_extra_bits filter (start.go AcceptsDP)
				u8* r = dps_frame.data() + 4 + (u64)HENRY_DP_REC_SIZE * n;
				PutU64(r, a.first_index + kind);
				memcpy(r + 8, p, 16);       // x[0..1]
				memcpy(r + 24, p + 16, 24); // d[0..2]
				r[48] = (u8)a.type;
				r[49] = r[50] = r[51] = 0;
				PutU32(r + 52, st.restarts[kind]);
				n++;
			}
			if (n)
			{
				PutU32(dps_frame.data(), n);
				if (!SendFrame(FRAME_DPS, dps_frame.data(), 4 + HENRY_DP_REC_SIZE * n))
					return Fatal("write to stdout failed");
			}
			done = gStopRequested || (a.max_blocks && st.blocks >= a.max_blocks);
		}

		u64 now = GetTickCount64();
		if (done || now - tm_stats >= STATS_INTERVAL_MS)
		{
			u64 dt = now - tm_stats;
			if (dt && iters > iters_at_stats)
				last_speed = (u32)(((iters - iters_at_stats) * jumps_per_iter) / (dt * 1000));
			if (!SendStats(st.blocks, iters * jumps_per_iter, last_speed))
				return Fatal("write to stdout failed");
			fprintf(stderr, "henry: %llu iterations, %u MKeys/s\n", (unsigned long long)st.blocks, last_speed);
			tm_stats = now;
			iters_at_stats = iters;
		}

		if ((done || since_ckpt >= a.ckpt_blocks) && (!have_ckpt || st.blocks > last_ckpt_blocks))
		{
			const char* cerr = SendCheckpoint(kang, a, m, st, dists4, l1s2_buf, lt_buf, blob);
			if (cerr)
				return Fatal("%s", cerr);
			since_ckpt = 0;
			have_ckpt = true;
			last_ckpt_blocks = st.blocks;
			// KernelB compares only the low 64 bits of d to find loops >2: a false
			// match (same d[0], different d) makes KernelC jump from a point that is
			// not point(d), and the walk goes on with point != point(d). A resume
			// recomputes the points from d, so the auditor's replay (which starts
			// from the checkpoint) would diverge from a GPU that just kept going.
			// Re-placing every kangaroo from the checkpoint's d makes the walk
			// continue from each CHECKPOINT exactly as a resume from it would.
			if (!done)
			{
				u64 t0 = GetTickCount64();
				if (!kang->HenryPlacePoints(dists4.data(), 4, err, sizeof(err)))
					return Fatal("%s", err);
				fprintf(stderr, "henry: points re-placed from checkpoint %llu in %llu ms\n",
					(unsigned long long)st.blocks, (unsigned long long)(GetTickCount64() - t0));
			}
		}
		if (done)
			break;
	}
	kang->HenryRelease();
	fprintf(stderr, "henry: stopped at %llu iterations\n", (unsigned long long)st.blocks);
	return 0;
}
