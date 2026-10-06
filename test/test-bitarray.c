#include <stdio.h>
#include <string.h>
#include <inttypes.h>

#include "test-harness.h"
#include "../bitarray.h"

int main(int argc      MRCAL_ATTRIBUTE((unused)),
         char* argv[]  MRCAL_ATTRIBUTE((unused)))
{
    const int Nbits = 350;

    const int Nwords = bitarray64_nwords(Nbits);
    uint64_t bitarray[Nwords];

    if(Nwords != 6)
    {
        printf("Mismatched Nwords\n");
        return 1;
    }
    uint64_t ref[6] = {};

    memset(bitarray, 0, Nwords*sizeof(uint64_t));

    bitarray64_set(      bitarray, 1);
    bitarray64_set_range(bitarray, 5, 30);
    bitarray64_clear(    bitarray, 6);
    bitarray64_set_range(bitarray, 60,4);
    ref[0] = 0xf0000007ffffffa2;

    bitarray64_set_range(bitarray, 64*1 + 60, 7);
    ref[1] = 0xf000000000000000;

    bitarray64_set_range(bitarray, 64*2 + 50, 100);
    ref[2] = 0xfffc000000000007;
    ref[3] = 0xffffffffffffffff;
    ref[4] = 0x00000000003fffff;

    bitarray64_set_range(bitarray, 64*5 + 0,  20);
    ref[5] = 0x00000000000fffff;

    if(false)
        for(int i=0; i<Nwords; i++)
            printf("word %d ref/computed/xor:\n0x%016"PRIx64"\n0x%016"PRIx64"\n0x%016"PRIx64"\n\n",
                   i,
                   ref[i],
                   bitarray[i],
                   ref[i] ^ bitarray[i]);

    confirm_eq_int(memcmp(ref, bitarray, Nwords*sizeof(uint64_t)), 0);

    confirm(!bitarray64_check(bitarray,64*2+50-1));
    confirm( bitarray64_check(bitarray,64*2+50));
    confirm( bitarray64_check(bitarray,64*2+50+100-1));
    confirm(!bitarray64_check(bitarray,64*2+50+100));

    confirm(!bitarray64_check_all_set  (bitarray,Nbits));
    confirm(!bitarray64_check_all_clear(bitarray,Nbits));

    for(int i=0; i<Nwords; i++) bitarray[i] = 0UL;

    confirm(!bitarray64_check_all_set(bitarray,Nbits));
    confirm(bitarray64_check_all_clear(bitarray,Nbits));

    // Set one-bit-past-the end. This is out-of-bounds and we should still be
    // all clear
    bitarray[Nbits/64] |= 1UL << (Nbits%64);
    confirm(!bitarray64_check_all_set(bitarray,Nbits));
    confirm( bitarray64_check_all_clear(bitarray,Nbits));

    // Set the last bit
    bitarray[Nbits/64] |= 1UL << ((Nbits%64)-1);
    confirm(!bitarray64_check_all_set(bitarray,Nbits));
    confirm(!bitarray64_check_all_clear(bitarray,Nbits));
    bitarray[Nbits/64] = 0;
    confirm(bitarray64_check_all_clear(bitarray,Nbits));

    // Set the first bit
    bitarray[0] |= 1UL;
    confirm(!bitarray64_check_all_set(bitarray,Nbits));
    confirm(!bitarray64_check_all_clear(bitarray,Nbits));

    TEST_FOOTER();
}
