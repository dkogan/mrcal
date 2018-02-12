#!/usr/bin/perl
use strict;
use warnings;

use feature ':5.10';
use IPC::Run 'run';
use FindBin '$Bin';
use Carp qw(cluck confess);
use autodie;

my @tests = ( "DISTORTION_NONE    extrinsics frames intrinsic-core intrinsic-distortions",
              "DISTORTION_NONE    extrinsics frames                intrinsic-distortions",
              "DISTORTION_NONE    extrinsics frames intrinsic-core",
              "DISTORTION_NONE    extrinsics frames",
              "DISTORTION_CAHVOR  extrinsics frames intrinsic-core intrinsic-distortions",
              "DISTORTION_CAHVOR  extrinsics frames                intrinsic-distortions",
              "DISTORTION_CAHVOR  extrinsics frames intrinsic-core",
              "DISTORTION_CAHVOR  extrinsics frames",
              "DISTORTION_OPENCV4 extrinsics frames intrinsic-core intrinsic-distortions",
              "DISTORTION_OPENCV4 extrinsics frames                intrinsic-distortions",
              "DISTORTION_OPENCV4 extrinsics frames intrinsic-core",
              "DISTORTION_OPENCV4 extrinsics frames",

              "DISTORTION_CAHVOR  frames intrinsic-core intrinsic-distortions",
              "DISTORTION_CAHVOR  frames                intrinsic-distortions",
              "DISTORTION_CAHVOR  frames intrinsic-core",
              "DISTORTION_CAHVOR  frames",

              "DISTORTION_CAHVOR  extrinsics intrinsic-core intrinsic-distortions",
              "DISTORTION_CAHVOR  extrinsics                intrinsic-distortions",
              "DISTORTION_CAHVOR  extrinsics intrinsic-core",
              "DISTORTION_CAHVOR  extrinsics",

              "DISTORTION_CAHVOR  intrinsic-core intrinsic-distortions",
              "DISTORTION_CAHVOR                 intrinsic-distortions",
              "DISTORTION_CAHVOR  intrinsic-core"
            );

for my $test (@tests)
{
    say $test;

    my $fd;
    open $fd, '-|', "$Bin/../test_gradients $test 2>/dev/null | vnl-filter --eval '{print ivar,imeasurement,error,error_relative}'";

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
