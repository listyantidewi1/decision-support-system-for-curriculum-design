echo.
echo [4/17] Verifying extracted skills...
python verify_skills.py
if errorlevel 1 goto :error

echo.
echo [5/17] Mapping knowledge to future job domains...
python future_weight_mapping.py
if errorlevel 1 goto :error

echo.
echo [6/17] Mapping skills to future job domains...
python future_weight_mapping.py --input_type skills
if errorlevel 1 goto :error

echo.
echo [7/17] Enriching outputs with job dates...
python enrich_with_dates.py
if errorlevel 1 goto :error

echo.
echo [8/17] Analyzing time trends (FDR-controlled, with stability)...
python skill_time_trend_analysis.py --only_hard --stability
if errorlevel 1 goto :error

echo.
echo [9/17] Generating initial competency proposals (with future + empirical trends)...
python generate_competencies.py
if errorlevel 1 goto :error

echo.
echo [10/17] Generating curriculum recommendations (with ablation + sensitivity)...
python recommendations.py --ablation --sensitivity
if errorlevel 1 goto :error

echo.
echo [11/17] Exporting gold set for labeling...
python export_gold_set.py --seed %SEED%
if errorlevel 1 goto :error

echo.
echo [12/17] Exporting for expert review (jobs, skills, knowledge)...
python export_for_review.py
if errorlevel 1 goto :error

echo.
echo [13/17] Exporting competencies for expert review...
python export_competencies_for_review.py
if errorlevel 1 goto :error

echo.
echo ===========================================================
echo  Stage A complete. All extraction and analysis outputs ready.
echo ===========================================================

echo.
echo ===========================================================
echo  STAGE B: Gold Set Labeling  (steps 14-17)
echo ===========================================================
echo.
echo  The gold labeling UI will now open in a new window.
echo  Label all items (skills, knowledge, domain mapping), then
echo  come back to THIS window and press any key to continue.
echo.
echo  URL: http://127.0.0.1:8001/?labeler_id=YOUR_NAME
echo.

start "Gold Labeling UI" cmd /c "cd /d D:\Projects\skill-extraction && python -m uvicorn gold_labeling_ui.app:app --port 8001"

timeout /t 3 >nul
start http://127.0.0.1:8001/?labeler_id=labeler1

echo  Waiting for you to finish labeling...
echo  Press any key when labeling is complete.
pause >nul

echo.
echo  Shutting down Gold Labeling UI...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8001 ^| findstr LISTENING') do (
    taskkill /PID %%a /F >nul 2>&1
)

echo.
echo [14/17] Merging gold labels (if multi-reviewer UI used)...
python merge_gold_labels.py
if errorlevel 1 echo [WARN] Merge gold labels had issues (may be single labeler)

echo.
echo [15/17] Evaluating extraction quality (using gold labels)...
python evaluate_extraction.py
if errorlevel 1 echo [WARN] Extraction evaluation had issues (check gold labels)

echo.
echo [16/17] Evaluating future-domain mapping (using gold labels)...
python evaluate_future_mapping.py
if errorlevel 1 echo [WARN] Future mapping evaluation had issues (check gold labels)

echo.
echo [17/17] Generating scientific analysis plots...
python plot_scientific_analysis.py
if errorlevel 1 echo [WARN] Scientific plots had issues (some inputs may be missing)

echo.
echo ===========================================================
echo  STAGE C: Expert Review
echo ===========================================================
echo.
echo  The expert review UI will now open in a new window.
echo  Review skills, knowledge, and competencies, then close
echo  the review UI window when finished.
echo.
echo  URL: http://127.0.0.1:8000/?reviewer_id=YOUR_NAME
echo.

start "Expert Review UI" cmd /c "cd /d D:\Projects\skill-extraction && python -m uvicorn review_ui.app:app --port 8000"

timeout /t 3 >nul
start http://127.0.0.1:8000/?reviewer_id=reviewer1

echo  Waiting for you to finish reviewing...
echo  Press any key when expert review is complete.
pause >nul

echo.
echo  Shutting down Expert Review UI...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8000 ^| findstr LISTENING') do (
    taskkill /PID %%a /F >nul 2>&1
)

echo.
echo ============================================================
echo  Phase 1 COMPLETE.
echo.
echo  Gold labeling results:
echo    DATA/labels/gold_skills.csv, gold_knowledge.csv
echo    results/extraction_evaluation_report.json
echo    results/future_mapping_evaluation_report.json
echo.
echo  Expert review feedback saved to:
echo    feedback_store/skill_feedback.csv
echo    feedback_store/knowledge_feedback.csv
echo    feedback_store/competency_feedback.csv
echo.
echo  Next: run  run_phase_2.bat  to process feedback and
echo        generate final outputs.
echo ============================================================
goto :end

:error
echo.
echo [ERROR] A step failed. Fix the error above and re-run.
pause
exit /b 1

:end
pause
