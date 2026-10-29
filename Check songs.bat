@echo off
rem Drag one or more songs (or a whole folder of songs) onto this file to check them for signs of AI generation.
rem The results stay in this window until you press a key.
if "%~1"=="" (
  echo.
  echo   To check a song, drag the song file ^(or a folder of songs^) onto the "Check songs" icon.
  echo   Supported: mp3, wav, flac, m4a, ogg, opus, aac.
  echo.
  pause
  exit /b
)
:next
if "%~1"=="" goto done
echo.
echo Checking %~nx1 ... ^(the first song takes about a minute while the detector loads^)
call "%~dp0run.bat" check "%~1"
shift
goto next
:done
echo.
echo A score is evidence, not proof. Never use it alone to accuse an artist.
pause
