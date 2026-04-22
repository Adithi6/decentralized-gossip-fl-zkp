const circomlibjs = require("circomlibjs");

async function main() {
  const secret = BigInt(process.argv[2]);

  const poseidon = await circomlibjs.buildPoseidon();
  const F = poseidon.F;

  const hash = poseidon([secret]);
  console.log(F.toString(hash));
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});