# These xxx markers are to be replaced by git_build_rpm
Name:           xxx
Version:        xxx

Release:        1%{?dist}
Summary:        Calibration tools

License:        proprietary
URL:            https://github.jpl.nasa.gov/maritime-robotics/mrcal/
Source0:        https://github.jpl.nasa.gov/maritime-robotics/mrcal/archive/%{version}.tar.gz#/%{name}-%{version}.tar.gz

BuildRequires: python36-numpy

# need this too for the /usr/include/numpy link. It points to the same place as
# the link in python36-numpy would point to, if they bothered to make one
BuildRequires: python2-numpy
# For the non-global parameters
BuildRequires: libdogleg-devel >= 0.15.3
BuildRequires: lapack-devel
BuildRequires: opencv-devel >= 3.2
BuildRequires: python36-devel
BuildRequires: python36-libs
BuildRequires: libminimath-devel
BuildRequires: mrbuild >= 1.4
BuildRequires: mrbuild-tools >= 1.4

# I need to run the python stuff in order to build the manpages
BuildRequires: numpysane >= 0.18-2
BuildRequires: gnuplotlib >= 0.28-3
BuildRequires: opencv-python36
BuildRequires: python36-scipy
BuildRequires: python36

Requires: numpysane >= 0.18-2
Requires: gnuplotlib >= 0.28-3
Requires: opencv-python36
Requires: python36-numpy >= 1.14.5
Requires: python36-scipy >= 0.18.1
Requires: python36
Requires: python36-ipython-console

%description
Calibration library
This is the C library and the python tools

%package        devel
Summary:        Development files for %{name}
Requires:       %{name}%{?_isa} = %{version}-%{release}

%description    devel
This is the headers and DSOs for other C applications to use this library

%prep
%setup -q

%build
make %{?_smp_mflags}

%install
rm -rf $RPM_BUILD_ROOT
%make_install

%files
%doc
%{_bindir}/*
%{_mandir}/*
%{_libdir}/*.so.*
%{python3_sitelib}/*

%files devel
%{_includedir}/*
%{_libdir}/*.so
