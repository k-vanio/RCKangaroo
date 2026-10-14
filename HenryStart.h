// henry (fork k-vanio): start distances derived from the server's seed.
// License: GPLv3, see "LICENSE.TXT" file

#pragma once

#include "defs.h"

// Sha256 hashes len bytes of msg into out (FIPS 180-4). Small public-domain
// style implementation kept inside the fork so the binary needs no new library.
void Sha256(const u8* msg, size_t len, u8 out[32]);

// HenryStartDistance is the C++ port of StartDistance in henry_cli
// (app/internal/kangaroo/start.go), bit for bit: the server recomputes any start
// point from (seed, range, type, global index, restart) to audit DPs and
// checkpoints (CLAUDE.md rule 16). Same distribution as the v4
// (GpuKang.cpp GenerateRndDistances):
//   TAME: uniform in [1, x32)              x32 = 2^(range-5); d = 0 re-derives
//   WILD: uniform in [0, 34*x32), even
// out = d as 3 little-endian limbs (always non-negative, < 2^171 for range <= 170).
// range must be in [6, 170].
void HenryStartDistance(const u8 seed[32], int range, int kang_type, u64 index, u32 restart, u64 out[3]);
