# Use the local mrbuild or the system mrbuild or tell the user how to download
# it

ifneq (,$(wildcard mrbuild/))
  MRBUILD_MK=mrbuild
  MRBUILD_BIN=mrbuild/bin
else ifneq (,$(wildcard /usr/include/mrbuild/Makefile.common.header))
  MRBUILD_MK=/usr/include/mrbuild
  MRBUILD_BIN=/usr/bin
else
  V      := 1.8
  SHA512 := 0d35dd2988a7ff74487d8cf9a259f208e4524d95ea392f063e31d05579ee7336a6ee4377b9a5e948694afbe46108da89e6a9afc0309a0b05851382a1f2fd038c
  URL   := https://github.com/dkogan/mrbuild/archive/refs/tags/v$V.tar.gz
  TARGZ := mrbuild-$V.tar.gz

  cmd := wget -O $(TARGZ) ${URL} && sha512sum --quiet --strict -c <(echo $(SHA512) $(TARGZ)) && tar xvfz $(TARGZ) && ln -fs mrbuild-$V mrbuild

  $(error mrbuild not found. Either 'apt install mrbuild', or if not possible, get it locally like this: '${cmd}')
endif
