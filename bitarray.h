#pragma once

// Simple array of bits, stored as 64-bit integers. Basic usage:
//
//   const int Nbits = ...;
//   const int Nwords = bitarray64_nwords(Nbits);
//   uint64_t bitarray[Nwords];
//   bitarray64_clear_all(bitarray, Nbits);

//   ...
//   bitarray64_set(bitarray, ibit);
//   ...
//   if(bitarray64_check(bitarray, ibit))
//   {
//       ...
//   }


#include <stdint.h>
#include <stdbool.h>

__attribute__((unused))
static inline int bitarray64_nwords(const int Nbits)
{
    // round up the number of 64-bit words required
    return (Nbits+63)/64;
}
__attribute__((unused))
static inline void bitarray64_set(uint64_t* bitarray, int ibit)
{
    bitarray[ibit/64] |= (1ul << (ibit % 64));
}
__attribute__((unused))
static inline void bitarray64_set_all(uint64_t* bitarray, const int Nbits)
{
    const int Nwords = bitarray64_nwords(Nbits);
    for(int i=0; i<Nwords; i++)
        bitarray[i] = ~0ULL;
}
__attribute__((unused))
static inline void bitarray64_clear_all(uint64_t* bitarray, const int Nbits)
{
    const int Nwords = bitarray64_nwords(Nbits);
    for(int i=0; i<Nwords; i++)
        bitarray[i] = 0ULL;
}
__attribute__((unused))
static inline void bitarray64_clear(uint64_t* bitarray, int ibit)
{
    bitarray[ibit/64] &= ~(1ul << (ibit % 64));
}
__attribute__((unused))
static inline bool bitarray64_check(const uint64_t* bitarray, int ibit)
{
    return bitarray[ibit/64] & (1ul << (ibit % 64));
}
__attribute__((unused))
static inline bool bitarray64_check_all_set(const uint64_t* bitarray, int Nbits)
{
    // Check the full words
    const int Nwords_full = Nbits/64; // rounds down
    for(int i=0; i<Nwords_full; i++)
        if(bitarray[i] != ~0ul)
            return false;
    const int Nbits_remaining = Nbits - Nwords_full*64;
    if(Nbits_remaining == 0)
        return true;
    // Check the last non-full word
    const uint64_t mask = (1UL << Nbits_remaining) - 1UL;
    return (bitarray[Nwords_full] & mask) == mask;
}
__attribute__((unused))
static inline bool bitarray64_check_all_clear(const uint64_t* bitarray, int Nbits)
{
    // Check the full words
    const int Nwords_full = Nbits/64; // rounds down
    for(int i=0; i<Nwords_full; i++)
        if(bitarray[i] != 0ul)
            return false;
    const int Nbits_remaining = Nbits - Nwords_full*64;
    if(Nbits_remaining == 0)
        return true;
    // Check the last non-full word
    const uint64_t mask = (1UL << Nbits_remaining) - 1UL;
    return (bitarray[Nwords_full] & mask) == 0;
}
__attribute__((unused))
static inline void bitarray64_set_range_oneword(uint64_t* word,
                                                int ibit0, int Nbits)
{
    *word |= ((1ul << Nbits) - 1) << ibit0;
}
__attribute__((unused))
static inline void bitarray64_set_range(uint64_t* bitarray,
                                        int ibit0, int Nbits)
{
    // The first chunk, up to the first word boundary
    int ibit_next_start_of_word = 64*(int)((ibit0+63)/64);

    int Nbits_remaining_in_word = ibit_next_start_of_word - ibit0;
    if(Nbits_remaining_in_word)
    {
        if(Nbits <= Nbits_remaining_in_word)
        {
            bitarray64_set_range_oneword(&bitarray[ibit0/64],
                                         ibit0%64,
                                         Nbits);
            return;
        }

        bitarray64_set_range_oneword(&bitarray[ibit0/64],
                                     ibit0%64,
                                     Nbits_remaining_in_word);

        ibit0 = ibit_next_start_of_word;
        Nbits -= Nbits_remaining_in_word;
    }

    // Next chunk starts at an even word boundary

    // Process any whole words
    while(Nbits >= 64)
    {
        bitarray[ibit0/64] = ~0ul;
        ibit0 += 64;
        Nbits -= 64;
    }

    // Last little bit
    if(Nbits)
        bitarray64_set_range_oneword(&bitarray[ibit0/64],
                                     0,
                                     Nbits);
}
