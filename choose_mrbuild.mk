# Use the local mrbuild or the system mrbuild or tell the user how to download
# it

ifneq (,$(wildcard mrbuild/))
  MRBUILD_MK=mrbuild
  MRBUILD_BIN=mrbuild/bin
else ifneq (,$(wildcard /usr/include/mrbuild/Makefile.common.header))
  MRBUILD_MK=/usr/include/mrbuild
  MRBUILD_BIN=/usr/bin
else
  V      := 1.5
  SHA512 := 8b322fa41351d6b7a3f43c1e05929e4dbe4630f0bde3c9a1b7d6e3138ef8749853afa49cc800ff1c64453ac203e1128cf1ddbbda829100a6d1aeb3eb9f6fb7b0

  URL   := https://github.com/dkogan/mrbuild/archive/refs/tags/v$V.tar.gz
  TARGZ := mrbuild-$V.tar.gz

  cmd := wget -O $(TARGZ) ${URL} && sha512sum --quiet --strict -c <(echo $(SHA512) $(TARGZ)) && tar xvfz $(TARGZ) && ln -fs mrbuild-$V mrbuild

  $(error mrbuild not found. Either 'apt install mrbuild', or if not possible, get it locally like this: '${cmd}')
endif
