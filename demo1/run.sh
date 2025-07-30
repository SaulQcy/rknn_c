cd build
rm -rf ./*
cmake ..
make
adb push demo1 /saul/

