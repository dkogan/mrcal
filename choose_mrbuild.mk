# Use the local mrbuild or the system mrbuild or tell the user how to download
# it

ifneq (,$(wildcard mrbuild/))
  MRBUILD_MK=mrbuild
  MRBUILD_BIN=mrbuild/bin
else ifneq (,$(wildcard /usr/include/mrbuild/Makefile.common.header))
  MRBUILD_MK=/usr/include/mrbuild
  MRBUILD_BIN=/usr/bin
else ifneq (,$(wildcard $(HOMEBREW_PREFIX)/include/mrbuild/Makefile.common.header))
  MRBUILD_MK=$(HOMEBREW_PREFIX)/include/mrbuild
  MRBUILD_BIN=$(HOMEBREW_PREFIX)/bin
else
  V      := 1.22
  SHA512 := b35c3cd7b2b56c61c70020e9e256baa8ec2bc1f73dd5ee3172dcaca2f0150c82951e9b5968d34c8d436f3b347dd38db85303316d7b1c00748fc3ea0365f92b57
  URL   := https://github.com/dkogan/mrbuild/archive/refs/tags/v$V.tar.gz
  TARGZ := mrbuild-$V.tar.gz

  cmd := curl -L -o $(TARGZ) ${URL} && echo "$(SHA512)  $(TARGZ)" | shasum -a 512 -c && tar xvfz $(TARGZ) && ln -fs mrbuild-$V mrbuild

  $(error mrbuild not found. Either 'apt install mrbuild' or 'brew install mrbuild' or get it locally like this: '${cmd}')
endif
