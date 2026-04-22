pragma circom 2.0.0;

include "circomlib/circuits/poseidon.circom";

template ClientAuth() {
    signal input secret;
    signal input commitment;

    component hash = Poseidon(1);
    hash.inputs[0] <== secret;

    commitment === hash.out;
}

component main = ClientAuth();