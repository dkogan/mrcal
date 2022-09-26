#!/usr/bin/perl
use strict;
use warnings;

use feature ':5.10';

my $usagemessage = "Usage: $0 frobnicate > frobnicate.pod";
if(@ARGV != 1)
{
    die "Need exactly one argument on the cmdline.\n$usagemessage";
}
my $path = $ARGV[0];
if( ! (-r $path && -x $path && -f $path) )
{
    die "Commandline argument must be an executable file\n$usagemessage";
}

# prepend ./ if no path given. I'm going to run this thing, so we need that
$path = "./$path" unless $path =~ m{^/};
my $helpstring = `$path --help`;
my $helpstring0 = $helpstring;


# I assume the following stucture. If the --help string doesn't fit this
# structure, I barf
#
# usage: frobnicator [-h] [--xxx XXX]
#                    a [b ...]
#
# Does thing
#
# Synopsis:
#
#   $ frobnicator xxx
#   ... blah blah
#
#   $ more examples ...
#
# long description that talks about stuff
# long description that talks about stuff
# long description that talks about stuff
#
# long description that talks about stuff
#
# long description that talks about stuff
#
# positional arguments:
#   a                     Does what a does
#
# optional arguments:
#   b                     Does what b does
# ...

# usage is the thing up to the first blank line
my ($usage) = $helpstring =~ m/(^usage.*?)\n\n/imsp
  or die "Couldn't parse out the usage; helpstring='$helpstring'";
$helpstring = ${^POSTMATCH};
my $helpstring1 = $helpstring;

# Then we have a one-line summary
my ($summary) = $helpstring =~ m/(^.*?)\n\n/p
  or die "Couldn't parse out the summary; helpstring='$helpstring'; helpstring0='$helpstring0'";
$helpstring = ${^POSTMATCH};
my $helpstring2 = $helpstring;

# Then the synopsis
my ($synopsis) = $helpstring =~
  m/ ^synopsis.*?\n\n          # synopsis title
     (                         # capture stuff
       (?:(?:[ \t] .+?)? \n)+  # a bunch of lines: empty or beginning with whitespace
     )                         # That's all I want
   /xpi
  or die "Couldn't parse out the synopsis; helpstring='$helpstring'; helpstring0='$helpstring0'; helpstring1='$helpstring1'";
$helpstring = ${^POSTMATCH};
my $helpstring3 = $helpstring;
$synopsis =~ s/\n*$//g; # cull trailing whitespace

# Now a description: everything until 'xxxx arguments'. I might not have a
# description at all. I might also not have any "arguments" sections.
my ($description, $post) = $helpstring =~ /(^.*?)(?:\n\n)?(\w+ arguments:?\n)/ips;
if( defined $description)
{
    $helpstring = $post . ${^POSTMATCH};
}
else
{
    # no arguments. Everything is a description.
    $description = $helpstring;
    $helpstring = '';
}
my $helpstring4 = $helpstring;


# Now the arguments
my @args;
while($helpstring !~ /^\s*$/)
{
    # I see "required arguments" or "optional arguments" or "options"
    my ($argument_kind) = $helpstring =~ /(^\w+ arguments|options):?\n\n?/pi
      or die "Couldn't parse out argument kind; helpstring='$helpstring'; helpstring0='$helpstring0'; helpstring1='$helpstring1'; helpstring2='$helpstring2'; helpstring3='$helpstring3'; helpstring4='$helpstring4'";
    $helpstring = ${^POSTMATCH};
    my $helpstring5 = $helpstring;

    my ($argument_what) = $helpstring =~ /(^.*?)(?:\n\n|\n$)/pis
      or die "Couldn't parse out argument what; helpstring='$helpstring'; helpstring0='$helpstring0'; helpstring1='$helpstring1'; helpstring2='$helpstring2'; helpstring3='$helpstring3'; helpstring4='$helpstring4'; helpstring5='$helpstring5'";
    $helpstring = ${^POSTMATCH};

    # I really should parse the table argparse puts out, but let's just finish
    # this as is for now
    push @args, [uc($argument_kind), $argument_what];
}

# Alrighty. I can now write out this thing
say "=head1 NAME\n";

my ($programname) = $path =~ m{([^/]+$)}
  or die "Couldn't parse out the program name";
say "$programname - $summary\n";

say "=head1 SYNOPSIS\n";

say linkify($synopsis);
say "";

if( $description )
{
    say "=head1 DESCRIPTION\n";

    say linkify($description);
    say "";
}

if(@args)
{
    say "=head1 OPTIONS\n";
    for my $arg (@args)
    {
        my ($kind,$what) = @$arg;
        $kind = "OPTIONAL ARGUMENTS" if $kind eq "OPTIONS";

        say "=head2 $kind\n";
        say linkify($what);
        say "";
    }
}



sub linkify
{
    $_[0] =~ s{https?://\S+}{L<${^MATCH}>}gps;
    return $_[0];
}

__END__

=head1 NAME

make-pod-from-help.pl - creates POD documentation from a commandline tool

=head1 SYNOPSIS

 $ ./make-pod-from-help.pl frobnicate > frobnicate.pod

=head1 DESCRIPTION

Python has a culture of ignoring standard ways of doing things, and building
their own everything from scratch for some reason. And as a result I need to
write this tool to give me manpages. This tool is a hack that works for THIS
project. A core assumption is that running the tool with C<--help> produces
fairly complete documentation, and I just need to massage it. My tools all have
a manpage-like docstring, use C<argparse> to document the options, and tell
C<argparse> to use the docstring in the C<--help> description.

This tool generates a POD, which can then be made into a manpage with
C<pod2man>.

Some details L<here|http://notes.secretsauce.net/notes/2018/10/07_generating-manpages-from-python-and-argparse.html>.

=head1 REQUIRED ARGUMENTS

=over

=item <inputprogram>

Tool that we're making a manpage for

=back

=head1 AUTHOR

Dima Kogan, C<< <dima@secretsauce.net> >>
