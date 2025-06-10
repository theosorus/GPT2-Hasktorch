#!/usr/bin/perl -w


# xml2txt.pl converts a Wikipedia XML corpus file (downloaded from 
# http://www.linguatools.org/tools/corpora/wikipedia-monolingual-corpora/)
# to raw text by discarding all XML mark-up and substituting the XML 
# entities (like &apos;) by their respective characters.
#
# Synopsis: xml2txt.pl [Options] INPUT OUTPUT
# -------------------------------------------
# Where INPUT is an unzipped Wikipedia XML corpus file, and OUTPUT is the
# raw text file that will be produced. The encoding of the output file 
# will be UTF-8.
# Options: -articles  The article mark-up is preserved 
#                      (<article name="...">...</article>).
#          -p         The paragraph mark-up is preserved (<p>...
#                      </p>).
#          -h         The headings mark-up is preserved (<h>...
#                      </h>).
#          -nomath    All content that is enclosed in math tags
#                      (<math>...</math>) is deleted.
#          -notables  All content that is enclosed in table tags
#                      (<table>...</table>) is deleted.
#          -nodisambig Articles marked as disambiguation articles are
#                      deleted.
#          -exclude-categories FILE   all articles that belong to one of the
#                      categories listed in FILE are ignored. FILE has one
#                      category name per line.
#          -only-categories FILE      only articles that belong to one of the
#                      categories listed in FILE are output. 
# (C) 2011-2018 Peter Kolb
# peter.kolb@linguatools.org
# License: CC0 1.0 Universal https://creativecommons.org/publicdomain/zero/1.0/legalcode
use strict;
use utf8;
use open ':utf8';
binmode STDOUT => ':utf8';
binmode STDIN => ':utf8';
binmode STDERR => ':utf8';


# parse options
my $input = "";
my $output = "";
my $option_nomath = 0;
my $option_notables = 0;
my $option_nodisambig = 0;
my $option_articles = 0;
my $option_p = 0;
my $option_h = 0;
my $exclude_cats = 0;
my $only_cats = 0;
my $cats_file = "";
if( $#ARGV < 1 ){
    printUsage();
    exit;
}
my $i; my $in = 0; 
my $i_is_n = -1; my $i_is_t = -1;
for($i = 0; $i <= $#ARGV; $i++){
    if( $ARGV[$i] eq "-nomath" ){
	$option_nomath = 1;
	print STDERR "deleting math formulas\n";
    }elsif( $ARGV[$i] eq "-notables" ){
	$option_notables = 1;
	print STDERR "deleting tables\n";
    }elsif( $ARGV[$i] eq "-nodisambig" ){
	$option_nodisambig = 1;
	print STDERR "deleting disambiguation articles\n";
    }elsif( $ARGV[$i] eq "-articles" ){
	$option_articles = 1;
	print STDERR "preserving articles boundaries\n";
    }elsif( $ARGV[$i] eq "-p" ){
	$option_p = 1;
	print STDERR "preserving paragraph boundaries\n";
    }elsif( $ARGV[$i] eq "-h" ){
	$option_h = 1;
	print STDERR "preserving heading tags\n";
    }elsif( $ARGV[$i] eq "-exclude-categories" ){ 
	$exclude_cats = 1; 
	$cats_file = $ARGV[$i + 1];					    
	$i++;
    }elsif( $ARGV[$i] eq "-only-categories" ){ 
	$only_cats = 1; 
	$cats_file = $ARGV[$i + 1];					    
	$i++;
    }elsif( $ARGV[$i] =~ /^-/ ){
	print STDERR "ERROR: unknown option \"$ARGV[$i]\"!\n";
	printUsage();
	exit;
    }else{
	if( $in == 0 ){
	    $input = $ARGV[$i];
	    $in = 1;
	}else{
	    $output = $ARGV[$i];
	}
    }
}


# checks
if( $input eq "" ){
    print "Error: no input file.\n";
    printUsage();
    exit;
}
if( $output eq "" ){
    print "Error: no output file.\n";
    printUsage();
    exit;
}
if( $exclude_cats == 1 && $only_cats == 1 ){
    print "Error: options -exclude-cats and -only-cats are mutually exclusive!\n";
    printUsage();
    exit;
}


# read categories file
my %CATS;
if( $cats_file ne "" ){
    read_categories($cats_file, \%CATS);
    my $c = keys %CATS;
    print "read $c categories from $cats_file\n";
}


# open input and output file
open(IN, "<$input") or die "Could not open input file $input: $!\n";
open(OUT, ">$output") or die "Could not open output file $output: $!\n";


my $filtered_disambig = 0;
my $filtered_cats = 0;
my $out = 0;
while(<IN>){
    if ( $_ =~ /<article name=\"(.+?)\">/ ){
	my $name = $1;
	# read article into scalar $a
	my $a = $_;
	while( $_ !~ /<\/article>/ && $_ ne "" ){
	    $_ = <IN>;
	    $a = $a . $_;
	}
	# substitute \n with &newline;
	$a =~ s/\n/&newline;/g;
	# no disambiguation articles with option -nodisambig
	if( $option_nodisambig == 1 ){
	    if( $a =~ /<disambiguation\/>/ ){
		$filtered_disambig++;
		next;
	    }
	}
	# categories
	if( $exclude_cats == 1 || $only_cats == 1 ){
	    if( $a =~ /<category name=\".*?\"\/>/ ){
		my @cats = $a =~ /<category name=\"(.*?)\"\/>/g;
		my $skip;
		if( $exclude_cats == 1 ){ $skip = 0; }
		if( $only_cats == 1 ){ $skip = 1; }
		foreach my $cat ( @cats ){
		    if( exists($CATS{$cat}) || exists($CATS{lc($cat)}) ){
			if( $exclude_cats == 1 ){
			    $skip = 1;
			    last;
			}
			if( $only_cats == 1 ){
			    $skip = 0;
			    last;
			}
		    }
		}
		if( $skip == 1 ){
		    $filtered_cats++;
		    next;
		}
	    }
	}
	# process $a
	if( $option_nomath == 1 ){
	    $a =~ s/<math>.+?<\/math>//g;
	}else{
	    $a =~ s/<\/?math>//g;
	}
	if( $option_notables == 1 ){
	    $a =~ s/<table>.+?<\/table>//g;
	}else{
	    $a =~ s/<\/?table>//g;
	}
	if( $option_articles == 0 ){
	    $a =~ s/<article name=\".+?\">//g;
	    $a =~ s/<\/article>//g;
	}
	if( $option_p == 0 ){
	    $a =~ s/<\/?p>//g;
	}else{
	    # mark-up has to stand on a line of its own
	    $a =~ s/<p>/&newline;<p>&newline;/g;
	    $a =~ s/<\/p>/&newline;<\/p>&newline;/g;
	}
	if( $option_h == 0 ){
	    $a =~ s/<\/?h>//g;
	}else{
	    # mark-up has to stand on a line of its own
	    $a =~ s/<h>/&newline;<h>&newline;/g;
	    $a =~ s/<\/h>/&newline;<\/h>&newline;/g;
	}
	$a =~ s/<!--.+?-->//g;
	$a =~ s/<\/?cell>/ /g;
	$a =~ s/<\/?content>//g;
	$a =~ s/<\/?wikipedia>//g;
	$a =~ s/<wikipedia lang=\".+?\">//g;
	$a =~ s/<redirect name=\".*?\"\/>//g;
	$a =~ s/<links_in name=\".*?\"\/>//g;
	$a =~ s/<links_out name=\".*?\"\/>//g;
	$a =~ s/<category name=\".*?\"\/>//g;
	$a =~ s/<crosslanguage_link language=\".+?\" name=\".+?\"\/>//g;
	$a =~ s/<link target=\".+?\">//g;
	$a =~ s/<\/link>//g;
	$a =~ s/<textlink name=\".+?\" freq=\"[0-9]+\"\/>//g;
	$a =~ s/<disambiguation\/>//g;
	# entities
	$a =~ s/&lt;/</g;
	$a =~ s/&gt;/>/g;
	$a =~ s/&quot;/\"/g;
	$a =~ s/&apos;/\'/g;
	$a =~ s/&amp;/&/g;
	# whitespace
	$a =~ s/&newline;\p{Z}+/&newline;/g;
	$a =~ s/\p{Z}+&newline;/&newline;/g;
	$a =~ s/(&newline;)+/&newline;/g;
	$a =~ s/\p{Z}+/ /g;
	# replace &newline; with \n
	$a =~ s/&newline;/\n/g;
	# output
	print OUT "$a";
	$out++;
    }
}


if( $option_nodisambig == 1 ){
    print "filtered $filtered_disambig disambiguation articles.\n";
}
if( $exclude_cats == 1 || $only_cats == 1 ){
    print "filtered $filtered_cats articles based on categories.\n";
}
print "wrote $out articles.\n";


################################ 
# read category list from file #
################################ 
sub read_categories {
    my $file = shift;
    my $h_ref = shift;


    open(EIN, "<$file") or die "ERROR: Could not open file $file\n";
    while(<EIN>){
	chomp;
	next if( /^\s*$/ );
	$h_ref->{$_} = 1;
    }
    close(EIN);
}


##############
# print help #
##############
sub printUsage {
    print STDERR <<END; 
xml2txt.pl [Options] INPUT OUTPUT
INPUT: unzipped Wikipedia XML corpus file (downloaded from
       www.linguatools.org/tools/corpora/wikipedia-monolingual-corpora/)
OUTPUT: raw text file (UTF-8)
Options: -articles   The article mark-up is preserved 
                     (<article name="...">...</article>).
	 -p          Paragraph mark-up is preserved (<p>...</p>).
         -h          Headings mark-up is preserved (<h>...</h>).
         -nomath     All content that is enclosed in math tags
                     (<math>...</math>) is deleted.
         -notables   All content that is enclosed in table tags
                     (<table>...</table>) is deleted.
         -nodisambig Articles that are marked as disambiguation
                     articles are deleted.
         -exclude-categories FILE   all articles that belong to one of the
                     categories listed in FILE are ignored. FILE has one
                     category name per line.
         -only-categories FILE      only articles that belong to one of the
                     categories listed in FILE are output. 
END
}
