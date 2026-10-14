// henry (fork k-vanio): distributed kangaroo mode for henry_cli (puzzle #135).
// License: GPLv3, see "LICENSE.TXT" file

#pragma once

// IsHenryInvocation: argv asks for -henry, -henry-info or -henry-selftest.
bool IsHenryInvocation(int argc, char* argv[]);

// HenryMain runs the mode and returns the process exit code. It never returns
// to the original v4 code path.
int HenryMain(int argc, char* argv[]);
