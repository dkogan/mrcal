%{
#include <stdio.h>
#include <stdlib.h>

extern int yylex();
extern int yyparse();
extern int yylineno;

void yyerror(const char *s);
%}

%union {
    float f;
    char* s;
}

%token <s> STRING
%token <f> FLOAT

%%
cameramodel:
    '{' keyvaluepairs '}' |
    '{' keyvaluepairs ',' '}' |

keyvaluepairs:
    keyvaluepairs ',' keyvalue |
    keyvalue;

keyvalue:
    key ':' value;

key:
    STRING { printf("got key '%s'\n", $1); }
    ;

value:
    STRING { printf("got value '%s'\n", $1); } |
    '[' floats ']'                             |
    '[' floats ',' ']'
    ;

floats:
    floats ',' float |
    float;

float:
    FLOAT {printf("got float %f\n", $1); }
    ;
%%

int main(int argc, char** argv)
{
    yyparse();

}

void yyerror(const char* errormessage)
{
    printf("Parser error on line %d: %s\n",
           yylineno,
           errormessage);
    exit(1);
}
