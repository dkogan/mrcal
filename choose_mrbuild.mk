# Use the local mrbuild or the system mrbuild or tell the user how to download
# it

ifneq (,$(wildcard mrbuild/))
  MRBUILD_MK=mrbuild
  MRBUILD_BIN=mrbuild/bin
else ifneq (,$(wildcard /usr/include/mrbuild/Makefile.common.header))
  MRBUILD_MK=/usr/include/mrbuild
  MRBUILD_BIN=/usr/bin
else
  V      := 1.10
  SHA512 := 6e2ced50d629bd4c97f25467cb56f7d2521f4322046271ffe610ec8abd7525dc436845872510f8504c8630109e31200429492673fe83f7c0d67ef56a0cb9602c
  URL   := https://github.com/dkogan/mrbuild/archive/refs/tags/v$V.tar.gz
  TARGZ := mrbuild-$V.tar.gz

  cmd := wget -O $(TARGZ) ${URL} && sha512sum --quiet --strict -c <(echo $(SHA512) $(TARGZ)) && tar xvfz $(TARGZ) && ln -fs mrbuild-$V mrbuild

  $(error mrbuild not found. Either 'apt install mrbuild', or if not possible, get it locally like this: '${cmd}')
endif
