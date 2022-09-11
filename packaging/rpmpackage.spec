Name:           mrcal
Version:        mrcal

Release:        1%{?dist}
Summary:        Calibration tools

License:        Apache-2.0
URL:            https://www.github.com/dkogan/mrcal/
Source0:        https://www.github.com/dkogan/mrcal/archive/%{version}.tar.gz#/%{name}-%{version}.tar.gz

BuildRequires: python36-numpy

# For the non-global parameters
BuildRequires: libdogleg-devel >= 0.15.4
BuildRequires: suitesparse-devel >= 5.4.0
BuildRequires: re2c >= 2

# I want suitesparse >= 5 at runtime. I'm using CHOLMOD_FUNCTION_DEFAULTS, which
# makes a reference to SuiteSparse_divcomplex
Requires: suitesparse >= 5.4.0

BuildRequires: lapack-devel
BuildRequires: python36-devel
BuildRequires: python36-libs

# I need to run the python stuff in order to build the manpages and to generate
# the npsp wrappers
BuildRequires: numpysane >= 0.35
BuildRequires: gnuplotlib >= 0.38
BuildRequires: opencv-python36
BuildRequires: python36-scipy
BuildRequires: python36

# some tests shell out to vnl-filter
BuildRequires: vnlog

# for minimath
BuildRequires: perl-List-MoreUtils

# for mrbuild
BuildRequires: chrpath

# for the parser
BuildRequires: re2c >= 2.0.3

Requires: numpysane >= 0.35
Requires: gnuplotlib >= 0.38
Requires: opencv-python36
Requires: python36-numpy >= 1.14.5
Requires: python36-scipy >= 0.18.1
Requires: python36-shapely
Requires: python36
Requires: python36-ipython-console
# for image_transformation_map(), not for plotting
Requires: python36-matplotlib
# for mrcal-stereo --viz stereo
Requires: python36-gl-image-display


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
make test-nosampling

%files
%doc
%{_bindir}/*
%{_mandir}/*
%{_libdir}/*.so.*
%{python3_sitelib}/*

%files devel
%{_includedir}/*
%{_libdir}/*.so
