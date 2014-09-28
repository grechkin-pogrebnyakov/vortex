#include "string.h"

#if !defined(__linux__)
//#include "stdafx.h"
#include <string>
#endif

#pragma warning(disable: 4996) //чтобы strcpy не выдавал warning


//#if !defined itoaxx && __itoaxx

char* itoaxx(int value, char*  str,int radix) {
    int  rem = 0;
    int  pos = 0;
    char ch  = '!' ;
    do
    {
        rem    = value % radix ;
        value /= radix;
        if ( 16 == radix )
        {
            if( rem >= 10 && rem <= 15 )
            {
                switch( rem )
                {
                    case 10:
                        ch = 'a' ;
                        break;
                    case 11:
                        ch ='b' ;
                        break;
                    case 12:
                        ch = 'c' ;
                        break;
                    case 13:
                        ch ='d' ;
                        break;
                    case 14:
                        ch = 'e' ;
                        break;
                    case 15:
                        ch ='f' ;
                        break;
                }
            }
        }
        if( '!' == ch )
        {
            str[pos++] = (char) ( rem + 0x30 );
        }
        else
        {
            str[pos++] = ch ;
        }
    }while( value != 0 );
    str[pos] = '\0' ;


    //char b[]="ABCD";
    size_t a = strlen(str);
    char *c;
    c=new(char);	
    
    strcpy(c,str);
    c[a]='\0';
    for(size_t i=0;i<a;i++)
      { c[a-1-i]=str[i];}
    strcpy(str,c);


    //return strrev(str);
    return str;
    //return c;
}

//#endif