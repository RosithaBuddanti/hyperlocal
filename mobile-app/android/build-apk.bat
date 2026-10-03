@echo off
set "JAVA_HOME=C:\Program Files\Android\Android Studio\jbr"
set "PATH=%JAVA_HOME%\bin;%PATH%"
echo Starting Gradle build with Java from Android Studio...
call gradlew.bat assembleDebug
echo Build finished with errorlevel %ERRORLEVEL%
