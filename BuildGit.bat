:: Run it in Git Shell. [ open Git Shell. copy this file in that directory and [this batch file name] <enter>
:: suppose your github has a system folder named CS415 and it has 3 components: abc-1.2, bbc-v1.0 and bbc-v2.0. To build the system you need only abc-1.2 and bbc-v2.0. The following batch script tries to build the system based on the given version.

@echo off
setlocal EnableExtensions EnableDelayedExpansion 

set url="https://github.com/anuraganands/GSGD"
git init
git remote add -f origin !url!

git push -u origin master

pause