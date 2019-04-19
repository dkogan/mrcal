# These xxx markers are to be replaced by git_build_rpm
Name:           xxx
Version:        xxx

Release:        1%{?dist}
Summary:        Calibration tools

License:        proprietary
URL:            https://github.jpl.nasa.gov/maritime-robotics/mrcal/
Source0:        https://github.jpl.nasa.gov/maritime-robotics/mrcal/archive/%{version}.tar.gz#/%{name}-%{version}.tar.gz

BuildRequires: numpy
BuildRequires: libdogleg-devel >= 0.15.2
BuildRequires: lapack-devel
BuildRequires: opencv-devel
BuildRequires: python34-devel
BuildRequires: python34-libs
BuildRequires: libminimath-devel
BuildRequires: mrbuild >= 0.61
BuildRequires: mrbuild-tools >= 0.61

# I need to run the python stuff in order to build the manpages
BuildRequires: numpysane
BuildRequires: gnuplotlib
BuildRequires: opencv-python
BuildRequires: scipy
BuildRequires: python34

Requires: numpysane
Requires: gnuplotlib
Requires: opencv-python
Requires: scipy
Requires: python34
Requires: python-ipython-console

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
