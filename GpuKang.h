// This file is a part of RCKangaroo software
// (c) 2024, RetiredCoder (RC)
// License: GPLv3, see "LICENSE.TXT" file
// https://github.com/RetiredC


#pragma once

#include "Ec.h"
#include "CallCubin.h"

#define STATS_WND_SIZE	16

struct EcJMP
{
	EcPoint p;
	EcInt dist;
};

//96bytes size
struct TPointPriv
{
	u64 x[4];
	u64 y[4];
	u64 priv[4];
};

class RCGpuKang
{
private:
	bool StopFlag;
	EcPoint PntToSolve;
	int Range; //in bits
	int DP; //in bits
	Ec ec;

	CriticalSection cr;
	std::vector<int> lsToRestart; //list of kangs to restart
	void DoRestartKangs();

	u32* DPs_out;
	TKparams Kparams;

	EcInt HalfRange;
	EcPoint PntHalfRange;
	EcPoint NegPntHalfRange;
	TPointPriv* RndPnts;
	EcJMP* EcJumps1;
	EcJMP* EcJumps2;
	EcJMP* EcJumps3;

	EcPoint PntWild;

	int cur_stats_ind;
	int SpeedStats[STATS_WND_SIZE];

	int Inv_DataSize;

	void GenerateRndDistances();
	bool Start();
	void Release();
#ifdef DEBUG_MODE
	int Dbg_CheckKangs();
#endif

	TCubinCall cc;
	bool Asm_CallGpuKernelAB();
public:
	int persistingL2CacheMaxSize;
	int CudaIndex; //gpu index in cuda
	int mpCnt;
	int KangCnt;
	int JumperInd;
	bool Failed;

	bool Is5xxx;
	int sm_inv_cnt; //number of SMs used for inverse calculation

	int CalcKangCnt();
	bool Prepare(EcPoint _PntToSolve, int _Range, int _DP, EcJMP* _EcJumps1, EcJMP* _EcJumps2, EcJMP* _EcJumps3);
	void Stop();
	void Execute();
	void ToRestartKangaroo(int KangInd);

	u32 dbg[256];

	int GetStatsSpeed();

	// henry (fork k-vanio): one herd, driven by HenryMain (Henry.cpp) from the
	// main thread; no DP database, no collision, no restart, no file.
	bool HenryMode = false; //set before Prepare
	bool HenryWild = false; //set before Prepare: the whole herd is WILD (else TAME)
	int GetBlockCnt() { return (int)Kparams.BlockCnt; }
	bool HenryPlacePoints(const u64* dists, int stride, char* err, int err_size);
	bool HenryStart(const u64* dists3, const u32* l1s2, const u64* loop_table, EcPoint _PntWild, char* err, int err_size);
	bool HenryIterate(u32* dp_cnt, u32** dps, char* err, int err_size);
	bool HenryReadState(u64* dists4, u32* l1s2, u64* loop_table, char* err, int err_size);
	bool HenryReadPoint(int kang_ind, EcPoint* pnt);
	void HenryRelease() { Release(); }
};
