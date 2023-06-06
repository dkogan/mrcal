# Use the local mrbuild or the system mrbuild or tell the user how to download
# it

ifneq (,$(wildcard mrbuild/))
  MRBUILD_MK=mrbuild
  MRBUILD_BIN=mrbuild/bin
else ifneq (,$(wildcard /usr/include/mrbuild/Makefile.common.header))
  MRBUILD_MK=/usr/include/mrbuild
  MRBUILD_BIN=/usr/bin
else
  MRBUILD_VER := 1.4
  URL         := https://github.com/dkogan/mrbuild/archive/refs/tags/v${MRBUILD_VER}.tar.gz

  cmd := wget -O v${MRBUILD_VER}.tar.gz ${URL} && sha512sum --quiet --strict -c mrbuild.checksums && tar xvfz v${MRBUILD_VER}.tar.gz && mv mrbuild-${MRBUILD_VER} mrbuild

  $(error mrbuild not found. Either 'apt install mrbuild', or if not possible, get it locally like this: '${cmd}')
endif
