How to compile libgpiod

1. clone the source code

```shell
git clone https://git.kernel.org/pub/scm/libs/libgpiod/libgpiod.git
```

2. set the C and CXX compilers

```shell
export CC=$RV1106_C_COMPILER
export CXX=$RV1106_CXX_COMPILER
```

3. compile

```shell
cd ~/code/libgpiod
./autogen.sh
./configure \
  --host=arm-rockchip830-linux-uclibcgnueabihf \
  --prefix=/usr \
  ac_cv_func_malloc_0_nonnull=yes \
  ac_cv_func_realloc_0_nonnull=yes
make clean
make
make DESTDIR=/home/saul/code/RV1106_rootfs install
```

4. copy the .so files to board

```shell
adb push libgpiod.so* /lib
```