rd /s /q docs
cd doc
call poetry run make.bat html
cd ..
move doc/build/html docs
