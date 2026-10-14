// henry (fork k-vanio): start distances derived from the server's seed.
// License: GPLv3, see "LICENSE.TXT" file

#include <string.h>
#include "HenryStart.h"

//////////////////////////////////////////////////////////////////////////////
// SHA-256 (FIPS 180-4)

static const u32 kSha256K[64] = {
	0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
	0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
	0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
	0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
	0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
	0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
	0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
	0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
};

static inline u32 Ror32(u32 x, int n) { return (x >> n) | (x << (32 - n)); }

static void Sha256Block(u32 h[8], const u8 b[64])
{
	u32 w[64];
	for (int i = 0; i < 16; i++)
		w[i] = ((u32)b[4 * i] << 24) | ((u32)b[4 * i + 1] << 16) | ((u32)b[4 * i + 2] << 8) | (u32)b[4 * i + 3];
	for (int i = 16; i < 64; i++)
	{
		u32 s0 = Ror32(w[i - 15], 7) ^ Ror32(w[i - 15], 18) ^ (w[i - 15] >> 3);
		u32 s1 = Ror32(w[i - 2], 17) ^ Ror32(w[i - 2], 19) ^ (w[i - 2] >> 10);
		w[i] = w[i - 16] + s0 + w[i - 7] + s1;
	}
	u32 a = h[0], bb = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
	for (int i = 0; i < 64; i++)
	{
		u32 S1 = Ror32(e, 6) ^ Ror32(e, 11) ^ Ror32(e, 25);
		u32 ch = (e & f) ^ (~e & g);
		u32 t1 = hh + S1 + ch + kSha256K[i] + w[i];
		u32 S0 = Ror32(a, 2) ^ Ror32(a, 13) ^ Ror32(a, 22);
		u32 mj = (a & bb) ^ (a & c) ^ (bb & c);
		u32 t2 = S0 + mj;
		hh = g; g = f; f = e; e = d + t1; d = c; c = bb; bb = a; a = t1 + t2;
	}
	h[0] += a; h[1] += bb; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
}

void Sha256(const u8* msg, size_t len, u8 out[32])
{
	u32 h[8] = { 0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19 };
	size_t full = len / 64;
	for (size_t i = 0; i < full; i++)
		Sha256Block(h, msg + 64 * i);
	u8 tail[128];
	size_t rem = len - 64 * full;
	memset(tail, 0, sizeof(tail));
	memcpy(tail, msg + 64 * full, rem);
	tail[rem] = 0x80;
	size_t tail_len = (rem + 1 + 8 <= 64) ? 64 : 128;
	u64 bits = (u64)len * 8;
	for (int i = 0; i < 8; i++)
		tail[tail_len - 1 - i] = (u8)(bits >> (8 * i));
	Sha256Block(h, tail);
	if (tail_len == 128)
		Sha256Block(h, tail + 64);
	for (int i = 0; i < 8; i++)
	{
		out[4 * i] = (u8)(h[i] >> 24);
		out[4 * i + 1] = (u8)(h[i] >> 16);
		out[4 * i + 2] = (u8)(h[i] >> 8);
		out[4 * i + 3] = (u8)h[i];
	}
}

//////////////////////////////////////////////////////////////////////////////
// StartDistance (start.go)

// Domain separation: the seed of a herd is used for nothing else.
static const char kStartDomain[] = "henry_cli/kangaroo/start/v1";

// startHash: 512 bits of SHA-256(domain || seed || index BE64 || restart BE32 || block),
// block = 2*attempt + {0,1}, as 8 little-endian limbs (limb 0 = least significant).
static void StartHash(const u8 seed[32], u64 index, u32 restart, int attempt, u64 v[8])
{
	const size_t dom_len = sizeof(kStartDomain) - 1;
	u8 msg[sizeof(kStartDomain) - 1 + 32 + 13];
	memcpy(msg, kStartDomain, dom_len);
	memcpy(msg + dom_len, seed, 32);
	u8* tail = msg + dom_len + 32;
	for (int i = 0; i < 8; i++)
		tail[i] = (u8)(index >> (56 - 8 * i));
	for (int i = 0; i < 4; i++)
		tail[8 + i] = (u8)(restart >> (24 - 8 * i));
	u8 buf[64];
	for (int blk = 0; blk < 2; blk++)
	{
		tail[12] = (u8)(2 * attempt + blk); // Go: byte(2*attempt + blk)
		Sha256(msg, sizeof(msg), buf + 32 * blk);
	}
	// buf is a big-endian 512-bit integer
	for (int i = 0; i < 8; i++)
	{
		u64 w = 0;
		for (int j = 0; j < 8; j++)
			w = (w << 8) | buf[56 - 8 * i + j];
		v[i] = w;
	}
}

// low 'bits' bits of v into out[3] (bits <= 192)
static void LowBits(const u64 v[8], int bits, u64 out[3])
{
	for (int i = 0; i < 3; i++)
	{
		int lo = 64 * i;
		if (bits >= lo + 64)
			out[i] = v[i];
		else if (bits > lo)
			out[i] = v[i] & ((1ull << (bits - lo)) - 1);
		else
			out[i] = 0;
	}
}

// (v >> shift) mod m, for m small
static u32 ShiftedMod(const u64 v[8], int shift, u32 m)
{
	int w = shift / 64, b = shift % 64;
	unsigned __int128 r = 0;
	for (int i = 7 - w; i >= 0; i--)
	{
		u64 lo = v[i + w];
		u64 hi = (i + w + 1 < 8) ? v[i + w + 1] : 0;
		u64 q = b ? ((lo >> b) | (hi << (64 - b))) : lo;
		r = ((r << 64) | q) % m;
	}
	return (u32)r;
}

void HenryStartDistance(const u8 seed[32], int range, int kang_type, u64 index, u32 restart, u64 out[3])
{
	// x32 = 2^(R-5) and 34*x32 = 17*2^(R-4) are both 2^s times a small number, so
	// "v mod max" of start.go needs no big integers:
	//   v mod 2^s           = low s bits of v
	//   v mod (17 * 2^s)    = ((v >> s) mod 17) * 2^s + (v mod 2^s)
	for (int attempt = 0; ; attempt++)
	{
		u64 v[8];
		StartHash(seed, index, restart, attempt, v);
		if (kang_type == TAME)
		{
			// TAME d = 0 would be the point at infinity (the v4 leaves it at (0,0),
			// off the curve); start.go re-derives with the next attempt.
			LowBits(v, range - 5, out);
			if (out[0] | out[1] | out[2])
				return;
			continue;
		}
		int s = range - 4;
		u32 r = ShiftedMod(v, s, 17);
		LowBits(v, s, out);
		// add r * 2^s (r < 17, s <= 166: fits in 192 bits, no overlap with the low part)
		int w = s / 64, b = s % 64;
		unsigned __int128 add = (unsigned __int128)r << b;
		u64 add_lo = (u64)add, add_hi = (u64)(add >> 64);
		out[w] |= add_lo;
		if (add_hi)
			out[w + 1] |= add_hi;
		out[0] &= ~1ull; // "must be even" (GpuKang.cpp:473)
		return;
	}
}
