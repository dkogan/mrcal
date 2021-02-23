# These xxx markers are to be replaced by git_build_rpm
Name:           xxx
Version:        xxx

Release:        1%{?dist}
Summary:        Calibration tools

License:        proprietary
URL:            https://www.github.com/dkogan/mrcal/
Source0:        https://www.github.com/dkogan/mrcal/archive/%{version}.tar.gz#/%{name}-%{version}.tar.gz

BuildRequires: python36-numpy

# For the non-global parameters
BuildRequires: libdogleg-devel >= 0.15.4
BuildRequires: suitesparse-devel >= 4.1.0
BuildRequires: lapack-devel
BuildRequires: python36-devel
BuildRequires: python36-libs

# I need to run the python stuff in order to build the manpages and to generate
# the npsp wrappers
BuildRequires: numpysane >= 0.29-1
BuildRequires: gnuplotlib >= 0.36
BuildRequires: opencv-python36
BuildRequires: python36-scipy
BuildRequires: python36

# some tests shell out to vnl-filter
BuildRequires: vnlog

# for minimath
BuildRequires: perl-List-MoreUtils

# for mrbuild
BuildRequires: chrpath

Requires: numpysane >= 0.29-1
Requires: gnuplotlib >= 0.36
Requires: opencv-python36
Requires: python36-numpy >= 1.14.5
Requires: python36-scipy >= 0.18.1
Requires: python36-shapely
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

%check
make test

%files
%doc
%{_bindir}/*
%{_mandir}/*
%{_libdir}/*.so.*
%{python3_sitelib}/*

%files devel
%{_includedir}/*
%{_libdir}/*.so
