%{
#include <stdio.h>
#include "cameramodel.tab.h"

extern int yylex();
%}

%option noyywrap
%option yylineno

%x SINGLEQUOTE
%x DOUBLEQUOTE


%%

<SINGLEQUOTE>[^']* {
    /* if we're in a quoted expression, read everything until the quote is
        closed */
    yylval.s = strdup(yytext);
    return STRING;
}
<DOUBLEQUOTE>[^"]* {
    /* if we're in a quoted expression, read everything until the quote is
        closed */
    yylval.s = strdup(yytext);
    return STRING;
}
<SINGLEQUOTE>['] {
    /* closing quote */
    BEGIN(INITIAL);
}
<DOUBLEQUOTE>\" {
    /* closing quote */
    BEGIN(INITIAL);
}

[{}:\[\],] {
    /* use these single chars directly */
    return yytext[0];
}

\' {
    /* if we see an opening quote, enter that mode */
    BEGIN(SINGLEQUOTE);
}
\" {
    /* if we see an opening quote, enter that mode */
    BEGIN(DOUBLEQUOTE);
}

#[^\n]* ; /* throw out comments */
[ \t\n] ; /* whitespace */

-?([0-9]+|[0-9]+\.[0-9]*|[0-9]*\.[0-9]+)([eE][+-]?[0-9]+)? {
  /* float */
  yylval.f = atof(yytext);
  return FLOAT;
}

. {
    fprintf(stderr, "Parser error on line %d: Got text '%s'\n", yylineno, yytext);
    exit(1);
}

%%
