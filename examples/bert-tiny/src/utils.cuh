#pragma once

#include <iostream>
#include <fstream>
#include <string>

#include "CKKS/Ciphertext.cuh"
#include "CKKS/Parameters.cuh"
#include "Transformer.cuh"

extern std::vector<FIDESlib::PrimeRecord> p64;
extern std::vector<FIDESlib::PrimeRecord> sp64;
extern FIDESlib::CKKS::Parameters params;

extern lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc;

inline const std::string root_dir = "/projectnb/he/seyda/FIDESlib/";

void prepare_gpu_context_bert(FIDESlib::CKKS::Context& cc_gpu,
                              const lbcrypto::KeyPair<lbcrypto::DCRTPoly>& keys,
                              size_t num_slots, size_t blockSize);

void create_cpu_context();

void prepare_cpu_context(const lbcrypto::KeyPair<lbcrypto::DCRTPoly>& keys,
                         size_t num_slots, size_t blockSize);   // <-- removed '?'
