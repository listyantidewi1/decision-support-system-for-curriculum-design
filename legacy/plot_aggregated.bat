@echo off
REM ============================================================
REM  Generate all plots from aggregated results
REM
REM  Prerequisites:
REM    - results_aggregated/ folder exists (from aggregate_results.py)
REM
REM  Outputs:
REM    - results_aggregated/figures/  (main + scientific plots)
REM ============================================================

cd /d D:\Projects\skill-extraction\

if not exist "results_aggregated" (
    echo [ERROR] results_aggregated folder not found.
    echo Run aggregate_results.py first, e.g.:
    echo   python aggregate_results.py --run_dirs results_run1 results_run2 --output_dir results_aggregated
    pause
    exit /b 1
)

echo.
echo [1/2] Generating main plots from results_aggregated...
python plot_generator.py --output_dir results_aggregated
if errorlevel 1 goto :error

echo.
echo [2/2] Generating scientific analysis plots...
python plot_scientific_analysis.py --output_dir results_aggregated
if errorlevel 1 echo [WARN] Scientific plots had issues (some inputs may be missing)

echo.
echo ============================================================
echo  Plots saved to results_aggregated\figures\
echo ============================================================
goto :end

:error
echo.
echo [ERROR] Plot generation failed.
pause
exit /b 1

:end
pause
