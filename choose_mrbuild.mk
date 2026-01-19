# Use the local mrbuild or the system mrbuild or tell the user how to download
# it

ifneq (,$(wildcard mrbuild/))
  MRBUILD_MK=mrbuild
  MRBUILD_BIN=mrbuild/bin
else ifneq (,$(wildcard /usr/include/mrbuild/Makefile.common.header))
  MRBUILD_MK=/usr/include/mrbuild
  MRBUILD_BIN=/usr/bin
else
  V      := 1.16
  SHA512 := ed10cc186e1c69152e0e3bb91384e115cdc2c11de08d07b9335f2e519c52d0d50cded9f6f43227475ffa80564734fb17f7d63b3ee8f6661bbca3ca5a29c7e994
  URL   := https://github.com/dkogan/mrbuild/archive/refs/tags/v$V.tar.gz
  TARGZ := mrbuild-$V.tar.gz

  cmd := wget -O $(TARGZ) ${URL} && sha512sum --quiet --strict -c <(echo $(SHA512) $(TARGZ)) && tar xvfz $(TARGZ) && ln -fs mrbuild-$V mrbuild

  $(error mrbuild not found. Either 'apt install mrbuild', or if not possible, get it locally like this: '${cmd}')
endif
