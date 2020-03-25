#!/usr/bin/perl
use strict;
use warnings;

use feature ':5.10';
use IPC::Run 'run';
use FindBin '$Bin';
use Carp qw(cluck confess);
use autodie;

my @tests = ( "LENSMODEL_PINHOLE extrinsics frames intrinsic-core intrinsic-distortions",
              "LENSMODEL_PINHOLE extrinsics frames                intrinsic-distortions",
              "LENSMODEL_PINHOLE extrinsics frames intrinsic-core",
              "LENSMODEL_PINHOLE extrinsics frames",
              "LENSMODEL_CAHVOR  extrinsics frames intrinsic-core intrinsic-distortions",
              "LENSMODEL_CAHVOR  extrinsics frames                intrinsic-distortions",
              "LENSMODEL_CAHVOR  extrinsics frames intrinsic-core",
              "LENSMODEL_CAHVOR  extrinsics frames",
              "LENSMODEL_OPENCV4 extrinsics frames intrinsic-core intrinsic-distortions",
              "LENSMODEL_OPENCV4 extrinsics frames                intrinsic-distortions",
              "LENSMODEL_OPENCV4 extrinsics frames intrinsic-core",
              "LENSMODEL_OPENCV4 extrinsics frames",

              "LENSMODEL_CAHVOR  frames intrinsic-core intrinsic-distortions",
              "LENSMODEL_CAHVOR  frames                intrinsic-distortions",
              "LENSMODEL_CAHVOR  frames intrinsic-core",
              "LENSMODEL_CAHVOR  frames",

              "LENSMODEL_CAHVOR  extrinsics intrinsic-core intrinsic-distortions",
              "LENSMODEL_CAHVOR  extrinsics                intrinsic-distortions",
              "LENSMODEL_CAHVOR  extrinsics intrinsic-core",
              "LENSMODEL_CAHVOR  extrinsics",

              "LENSMODEL_CAHVOR  intrinsic-core intrinsic-distortions",
              "LENSMODEL_CAHVOR                 intrinsic-distortions",
              "LENSMODEL_CAHVOR  intrinsic-core"
            );

for my $test (@tests)
{
    say $test;

    my $fd;
    open $fd, '-|', "$Bin/../test-gradients $test 2>/dev/null | vnl-filter --eval '{print ivar,imeasurement,error,error_relative}'";

    my $err_relative_max           = -1.0;
    my $err_absolute_corresponding = -1.0;
    my $ivar_corresponding;
    my $imeas_corresponding;

    while (<$fd>)
    {
        my ($ivar,$imeas,$err_absolute,$err_relative) = split;
        if ( abs($err_absolute) > 1e-8 && $err_relative > $err_relative_max )
        {
            $err_relative_max           = $err_relative;
            $err_absolute_corresponding = $err_absolute;
            $ivar_corresponding         = $ivar;
            $imeas_corresponding        = $imeas;
        }
    }

    close $fd;
    say "var/meas: $ivar_corresponding/$imeas_corresponding err_relative_max: $err_relative_max: err_absolute: $err_absolute_corresponding";
}
