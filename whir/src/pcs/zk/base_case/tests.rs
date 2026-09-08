//! Standalone Construction 7.2 tests over a directly-committed source.
//!
//! A directly-committed width-one codeword is the trivial virtual fold.
//! The closure-based source abstraction collapses to plain openings.

use alloc::vec;
use alloc::vec::Vec;

use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::{CanObserve, DuplexChallenger, FieldChallenger};
use p3_commit::{ExtensionMmcs, Mmcs};
use p3_dft::Radix2DFTSmallBatch;
use p3_field::extension::BinomialExtensionField;
use p3_field::{Field, PrimeCharacteristicRing, dot_product};
use p3_matrix::Dimensions;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use p3_zk_codes::{ZkEncoding, ZkEncodingWithRandomness};
use proptest::prelude::*;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

use super::*;
use crate::pcs::proof::{QueryOpenings, SharedProofOpening};
use crate::pcs::zk::committer::FoldedRsCode;
use crate::pcs::zk::mask::{FreshMaskBatching, MaskCodeShape, MaskGroupShape};
use crate::pcs::zk::proof::{BaseCaseZkProof, BlindedMask};

type F = BabyBear;
type EF = BinomialExtensionField<F, 4>;
type Perm = Poseidon2BabyBear<16>;
type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
type MyChallenger = DuplexChallenger<F, Perm, 16, 8>;
type PackedF = <F as Field>::Packing;
type MyMmcs = MerkleTreeMmcs<PackedF, PackedF, MyHash, MyCompress, 2, 8>;
type MyDft = Radix2DFTSmallBatch<F>;

fn setup() -> (
    MyChallenger,
    MyChallenger,
    ExtensionMmcs<F, EF, MyMmcs>,
    MyDft,
) {
    let perm = Perm::new_from_rng_128(&mut SmallRng::seed_from_u64(1));
    let prover_challenger = MyChallenger::new(perm.clone());
    let verifier_challenger = MyChallenger::new(perm.clone());
    let mmcs = MyMmcs::new(MyHash::new(perm.clone()), MyCompress::new(perm), 0);
    (
        prover_challenger,
        verifier_challenger,
        ExtensionMmcs::new(mmcs),
        MyDft::default(),
    )
}

/// Which committed oracle a run should desynchronize from its reveals.
///
/// The reveals and target are always built from the original messages, so
/// the joint target check still closes; only the committed codeword differs,
/// so verification proceeds to the spot-check equation under test.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Tamper {
    /// Commit and reveal the same messages (an honest run).
    None,
    /// Commit a source codeword that differs from the revealed message.
    SourceMessage,
    /// Commit a mask codeword that differs from the revealed message.
    MaskMessage,
}

/// One standalone run, returning everything a check needs.
#[allow(clippy::type_complexity)]
fn honest_run(
    seed: u64,
    num_masks: usize,
    pow_bits: usize,
    tamper: Tamper,
    source_override: Option<Vec<EF>>,
) -> (
    BaseCaseZkConfig<F>,
    ExtensionMmcs<F, EF, MyMmcs>,
    BaseCaseZkProof<F, EF, MyMmcs>,
    Vec<EF>,
    Vec<Vec<EF>>,
    Vec<<MyMmcs as Mmcs<F>>::Commitment>,
    EF,
    <MyMmcs as Mmcs<F>>::Commitment,
    MyChallenger,
) {
    honest_run_with_batching(
        seed,
        num_masks,
        pow_bits,
        tamper,
        source_override,
        FreshMaskBatching::Separate,
        None,
    )
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn honest_run_with_batching(
    seed: u64,
    num_masks: usize,
    pow_bits: usize,
    tamper: Tamper,
    source_override: Option<Vec<EF>>,
    batching: FreshMaskBatching,
    groups_override: Option<Vec<MaskGroupShape>>,
) -> (
    BaseCaseZkConfig<F>,
    ExtensionMmcs<F, EF, MyMmcs>,
    BaseCaseZkProof<F, EF, MyMmcs>,
    Vec<EF>,
    Vec<Vec<EF>>,
    Vec<<MyMmcs as Mmcs<F>>::Commitment>,
    EF,
    <MyMmcs as Mmcs<F>>::Commitment,
    MyChallenger,
) {
    let mut rng = SmallRng::seed_from_u64(seed);
    let (mut prover_challenger, verifier_challenger, extension_mmcs, dft) = setup();

    // Source: message 8, randomness 3, domain 16, committed directly.
    let code = FoldedRsCode::<F>::new(8, 3, 16);
    let source_message: Vec<EF> =
        source_override.unwrap_or_else(|| (0..code.message_len).map(|_| rng.random()).collect());
    let source_randomness: Vec<EF> = (0..code.randomness_len).map(|_| rng.random()).collect();
    // Under SourceMessage the committed codeword encodes a shifted message,
    // while the reveals and target keep the original.
    let mut committed_source = source_message.clone();
    if tamper == Tamper::SourceMessage {
        committed_source[0] += EF::ONE;
    }
    let source_codeword = code.encode_column(&dft, &committed_source, &source_randomness);
    let (source_commitment, source_data) = extension_mmcs.commit_matrix(source_codeword);

    // Carried masks as width-one groups with their own RS-ZK codes.
    let mask_groups: Vec<MaskGroupShape> = groups_override.unwrap_or_else(|| {
        (0..num_masks)
            .map(|i| MaskGroupShape {
                shape: MaskCodeShape::new(4 + i, 2, 1),
                width: 1,
            })
            .collect()
    });
    let mut mask_messages = Vec::new();
    let mut mask_randomness = Vec::new();
    let mut mask_covectors = Vec::new();
    let mut mask_commitments = Vec::new();
    let mut mask_data = Vec::new();
    for (i, group) in mask_groups.iter().enumerate() {
        let encoding = group.shape.encoding::<EF>();
        let mut messages = Vec::new();
        let mut randomness = Vec::new();
        let mut committed = Vec::new();
        for column in 0..group.width {
            let message = encoding.sample_message(&mut rng);
            randomness.push(encoding.sample_randomness(&mut rng));
            let mut committed_message = message.clone();
            if tamper == Tamper::MaskMessage && i == 0 && column == 0 {
                committed_message[0] += EF::ONE;
            }
            committed.push(committed_message);
            messages.push(message);
            mask_covectors.push(
                (0..group.shape.message_len)
                    .map(|_| rng.random())
                    .collect::<Vec<EF>>(),
            );
        }
        let codeword = encoding.encode_batch_with_randomness(&committed, &randomness);
        let (commitment, data) = extension_mmcs.commit_matrix(codeword);
        mask_messages.push(messages);
        mask_randomness.push(randomness);
        mask_commitments.push(commitment);
        mask_data.push(data);
    }

    // Joint relation and its honest target.
    let source_covector: Vec<EF> = (0..code.message_len).map(|_| rng.random()).collect();
    let mut target = dot_product::<EF, _, _>(
        source_message.iter().copied(),
        source_covector.iter().copied(),
    );
    for (message, covector) in mask_messages.iter().flatten().zip(&mask_covectors) {
        target += dot_product::<EF, _, _>(message.iter().copied(), covector.iter().copied());
    }

    let config = BaseCaseZkConfig {
        fresh_mask_batching: batching,
        code,
        mask_groups,
        num_queries: 4,
        mask_queries: 2,
        pow_bits,
    };
    let prover = BaseCaseZkProver {
        config: &config,
        extension_mmcs: &extension_mmcs,
    };

    let mut mask_offset = 0;
    let witnesses: Vec<MaskGroupWitness<'_, F, EF, MyMmcs>> = mask_messages
        .iter()
        .zip(&mask_randomness)
        .zip(&mask_data)
        .map(|((messages, randomness), data)| {
            let covectors = &mask_covectors[mask_offset..mask_offset + messages.len()];
            mask_offset += messages.len();
            MaskGroupWitness {
                messages,
                randomness,
                covectors,
                data,
            }
        })
        .collect();

    let proof = prover.prove(
        &dft,
        &source_message,
        &source_randomness,
        &source_covector,
        &witnesses,
        |positions| {
            QueryOpenings::Extension(SharedProofOpening::open(
                &extension_mmcs,
                positions,
                &source_data,
            ))
        },
        &mut prover_challenger,
        &mut rng,
    );

    if tamper == Tamper::None {
        let mut checked_challenger = verifier_challenger.clone();
        let verifier = BaseCaseZkVerifier {
            config: &config,
            extension_mmcs: &extension_mmcs,
        };
        verifier
            .verify(
                &proof,
                &source_covector,
                &mask_covectors,
                &mask_commitments,
                target,
                |positions, opening| {
                    let QueryOpenings::Extension(opening) = opening else {
                        unreachable!()
                    };
                    opening
                        .verify(
                            &extension_mmcs,
                            &source_commitment,
                            &[Dimensions {
                                height: config.code.domain_size,
                                width: 1,
                            }],
                            positions,
                        )
                        .map_err(|_| BaseCaseZkError::SourceOpeningsRejected)?;
                    Ok(opening.rows.iter().map(|row| row[0]).collect())
                },
                &mut checked_challenger,
            )
            .unwrap();
        let prover_next: EF = prover_challenger.sample_algebra_element();
        let verifier_next: EF = checked_challenger.sample_algebra_element();
        assert_eq!(prover_next, verifier_next, "base-case transcript parity");
    }

    (
        config,
        extension_mmcs,
        proof,
        source_covector,
        mask_covectors,
        mask_commitments,
        target,
        source_commitment,
        verifier_challenger,
    )
}

/// Runs the verifier against a (possibly tampered) standalone proof.
#[allow(clippy::too_many_arguments)]
fn verify_run(
    config: &BaseCaseZkConfig<F>,
    extension_mmcs: &ExtensionMmcs<F, EF, MyMmcs>,
    proof: &BaseCaseZkProof<F, EF, MyMmcs>,
    source_covector: &[EF],
    mask_covectors: &[Vec<EF>],
    mask_commitments: &[<MyMmcs as Mmcs<F>>::Commitment],
    target: EF,
    source_commitment: &<MyMmcs as Mmcs<F>>::Commitment,
    mut challenger: MyChallenger,
) -> Result<(), BaseCaseZkError> {
    let verifier = BaseCaseZkVerifier {
        config,
        extension_mmcs,
    };
    let dims = [Dimensions {
        height: config.code.domain_size,
        width: 1,
    }];
    verifier.verify(
        proof,
        source_covector,
        mask_covectors,
        mask_commitments,
        target,
        |positions, openings| {
            let QueryOpenings::Extension(opening) = openings else {
                return Err(BaseCaseZkError::SourceOpeningsRejected);
            };
            opening
                .verify(extension_mmcs, source_commitment, &dims, positions)
                .map_err(|_| BaseCaseZkError::SourceOpeningsRejected)?;
            // Width-one source: the folded value is the single opened column.
            Ok(opening.rows.iter().map(|row| row[0]).collect())
        },
        &mut challenger,
    )
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(16))]

    #[test]
    fn prop_base_case_completeness(seed in any::<u64>(), num_masks in 0usize..=3) {
        // Honest standalone runs accept across mask counts and seeds,
        // with and without grinding.
        let pow_bits = (seed % 2) as usize * 4;
        let (config, mmcs, proof, w, u, commits, target, source, challenger) =
            honest_run(seed, num_masks, pow_bits, Tamper::None, None);
        prop_assert!(
            verify_run(&config, &mmcs, &proof, &w, &u, &commits, target, &source, challenger)
                .is_ok()
        );
    }
}

#[test]
fn base_case_rejects_wrong_target() {
    // The joint linear identity pins the carried target.
    let (config, mmcs, proof, w, u, commits, target, source, challenger) =
        honest_run(3, 2, 0, Tamper::None, None);
    let err = verify_run(
        &config,
        &mmcs,
        &proof,
        &w,
        &u,
        &commits,
        target + EF::ONE,
        &source,
        challenger,
    )
    .unwrap_err();
    assert_eq!(err, BaseCaseZkError::TargetCheckFailed);
}

#[test]
fn base_case_rejects_tampered_blinded_message() {
    // The shifted reveal enters the joint linear identity directly,
    // so the target check fires before any spot check runs.
    let (config, mmcs, mut proof, w, u, commits, target, source, challenger) =
        honest_run(4, 1, 0, Tamper::None, None);
    proof.blinded_message[0] += EF::ONE;
    let err = verify_run(
        &config, &mmcs, &proof, &w, &u, &commits, target, &source, challenger,
    )
    .unwrap_err();
    assert_eq!(err, BaseCaseZkError::TargetCheckFailed);
}

#[test]
fn base_case_rejects_tampered_blinded_randomness() {
    // Invariant: a randomness-reveal shift is caught after the target check.
    //
    //     target identity   ->  unaffected (randomness is not in it)
    //     reveal absorbed   ->  before the spot positions are drawn
    //     failure           ->  diverged openings or encoding equation
    let (config, mmcs, mut proof, w, u, commits, target, source, challenger) =
        honest_run(5, 1, 0, Tamper::None, None);
    proof.blinded_randomness[0] += EF::ONE;
    let err = verify_run(
        &config, &mmcs, &proof, &w, &u, &commits, target, &source, challenger,
    )
    .unwrap_err();
    // The randomness reveal is absorbed before the spot positions are drawn.
    //   verifier samples positions != the ones the proof was built for
    //   -> the source multiproof opens leaves the verifier never asked for
    //   -> index binding rejects the opening
    assert_eq!(err, BaseCaseZkError::SourceOpeningsRejected);
}

#[test]
fn base_case_rejects_tampered_mask_reveal() {
    // The shifted mask reveal enters the joint linear identity directly,
    // so the target check fires before any spot check runs.
    let (config, mmcs, mut proof, w, u, commits, target, source, challenger) =
        honest_run(6, 2, 0, Tamper::None, None);
    proof.blinded_masks[1].message[0] += EF::ONE;
    let err = verify_run(
        &config, &mmcs, &proof, &w, &u, &commits, target, &source, challenger,
    )
    .unwrap_err();
    assert_eq!(err, BaseCaseZkError::TargetCheckFailed);
}

#[test]
fn reveal_digest_binds_every_segment_and_mask_order() {
    let (_, _, mmcs, _) = setup();
    let message = vec![EF::from_u32(1), EF::from_u32(2), EF::from_u32(3)];
    let randomness = vec![EF::from_u32(4), EF::from_u32(5)];
    let masks = vec![
        BlindedMask {
            message: vec![EF::from_u32(6), EF::from_u32(7)],
            randomness: vec![EF::from_u32(8)],
        },
        BlindedMask {
            message: vec![EF::from_u32(9)],
            randomness: vec![EF::from_u32(10), EF::from_u32(11)],
        },
    ];
    let digest = |message: &[EF], randomness: &[EF], masks: &[BlindedMask<EF>]| {
        postcard::to_allocvec(&super::reveal_digest::reveal_digest(
            &mmcs, message, randomness, masks,
        ))
        .unwrap()
    };
    let expected = digest(&message, &randomness, &masks);

    let mut changed_message = message.clone();
    changed_message[0] += EF::ONE;
    assert_ne!(digest(&changed_message, &randomness, &masks), expected);

    let mut changed_randomness = randomness.clone();
    changed_randomness[0] += EF::ONE;
    assert_ne!(digest(&message, &changed_randomness, &masks), expected);

    let mut changed_mask_message = masks.clone();
    changed_mask_message[0].message[0] += EF::ONE;
    assert_ne!(
        digest(&message, &randomness, &changed_mask_message),
        expected
    );

    let mut changed_mask_randomness = masks.clone();
    changed_mask_randomness[1].randomness[0] += EF::ONE;
    assert_ne!(
        digest(&message, &randomness, &changed_mask_randomness),
        expected
    );

    let mut reordered_masks = masks.clone();
    reordered_masks.swap(0, 1);
    assert_ne!(digest(&message, &randomness, &reordered_masks), expected);
}

#[test]
fn base_case_rejects_unbound_source_reveal() {
    // The committed source encodes a shifted message; the reveals and
    // target keep the original.
    //
    //     target check  ->  passes (reveals and target agree)
    //     source check  ->  Enc(f*, r*)(z) != g(z) + gamma * f(z)
    //
    // This is the only branch that ties f* to the committed source.
    let (config, mmcs, proof, w, u, commits, target, source, challenger) =
        honest_run(30, 1, 0, Tamper::SourceMessage, None);
    let err = verify_run(
        &config, &mmcs, &proof, &w, &u, &commits, target, &source, challenger,
    )
    .unwrap_err();
    // The committed source genuinely differs from the reveal, so the source
    // spot check fails. The failing position is fixed by the test seed.
    assert_eq!(err, BaseCaseZkError::SourceSpotCheckFailed { position: 1 });
}

#[test]
fn base_case_rejects_unbound_mask_reveal() {
    // The committed mask encodes a shifted message; the reveals and
    // target keep the original.
    //
    //     target check  ->  passes (reveals and target agree)
    //     source check  ->  passes (source is honest here)
    //     mask check    ->  Enc(xi*, r*)(y) != s'(y) + gamma * xi(y)
    //
    // This is the only branch that ties xi* to the committed mask.
    let (config, mmcs, proof, w, u, commits, target, source, challenger) =
        honest_run(31, 1, 0, Tamper::MaskMessage, None);
    let err = verify_run(
        &config, &mmcs, &proof, &w, &u, &commits, target, &source, challenger,
    )
    .unwrap_err();
    // The committed mask genuinely differs from the reveal, so the mask
    // spot check fails in group 0. The failing position is fixed by the seed.
    assert_eq!(
        err,
        BaseCaseZkError::MaskSpotCheckFailed {
            group: 0,
            position: 0
        }
    );
}

#[test]
fn base_case_reveals_are_one_time_padded() {
    // Invariant (Lemma 7.3): the source reveal is a one-time pad.
    //
    //     reveal = g~ + gamma * secret
    //
    // The fresh mask g~ and the challenge gamma do not depend on the secret.
    // Two runs share the fresh-mask stream but differ in the secret at
    // coordinate 0 only:
    //
    //     reveal_a - reveal_b = gamma * (secret_a - secret_b)
    //     recovered g~        = reveal - gamma * secret  (equal across runs)
    let seed = 7;
    let message_len = 8_usize;

    // Two secrets differing only at coordinate 0.
    let secret_a: Vec<EF> = (0..message_len).map(|j| EF::from_u64(j as u64)).collect();
    let mut secret_b = secret_a.clone();
    secret_b[0] += EF::ONE;

    let (_, _, proof_a, ..) = honest_run(seed, 1, 0, Tamper::None, Some(secret_a.clone()));
    let (_, _, proof_b, ..) = honest_run(seed, 1, 0, Tamper::None, Some(secret_b.clone()));

    // The fresh mask is witness-independent, so its commitment matches.
    assert_eq!(
        postcard::to_allocvec(&proof_a.fresh_main_commitment).unwrap(),
        postcard::to_allocvec(&proof_b.fresh_main_commitment).unwrap(),
    );

    // Recover gamma from the one coordinate where the secrets differ:
    //
    //     reveal_a[0] - reveal_b[0] = gamma * (secret_a[0] - secret_b[0])
    let gamma = (proof_a.blinded_message[0] - proof_b.blinded_message[0])
        * (secret_a[0] - secret_b[0]).inverse();
    assert!(!gamma.is_zero(), "the pad must mix the secret");

    // The pad is present: the reveal is not the bare secret.
    assert_ne!(proof_a.blinded_message, secret_a);

    // Both runs recover the same fresh mask g~ = reveal - gamma * secret,
    // which holds only if the reveal is exactly g~ + gamma * secret.
    for i in 0..message_len {
        let g_from_a = proof_a.blinded_message[i] - gamma * secret_a[i];
        let g_from_b = proof_b.blinded_message[i] - gamma * secret_b[i];
        assert_eq!(g_from_a, g_from_b, "recovered fresh mask differs at {i}");
    }
}

fn mixed_groups() -> Vec<MaskGroupShape> {
    vec![
        MaskGroupShape {
            shape: MaskCodeShape::new(4, 2, 1),
            width: 2,
        },
        MaskGroupShape {
            shape: MaskCodeShape::new(7, 2, 1),
            width: 1,
        },
        MaskGroupShape {
            shape: MaskCodeShape::new(5, 2, 1),
            width: 3,
        },
        MaskGroupShape {
            shape: MaskCodeShape::new(4, 2, 1),
            width: 1,
        },
    ]
}

#[test]
fn fresh_batches_preserve_group_and_column_order() {
    let groups = mixed_groups();
    let batches = FreshMaskBatching::SameHeight.batches(&groups);
    assert_eq!(batches.len(), 2);
    assert_eq!(batches[0].group_indices, vec![0, 2, 3]);
    assert_eq!((batches[0].domain_size, batches[0].width), (16, 6));
    assert_eq!(batches[1].group_indices, vec![1]);
    assert_eq!((batches[1].domain_size, batches[1].width), (32, 1));
    let separate = FreshMaskBatching::Separate.batches(&groups);
    assert_eq!(separate.len(), 4);
    for (index, batch) in separate.iter().enumerate() {
        assert_eq!(batch.group_indices, vec![index]);
        assert_eq!(batch.width, groups[index].width);
    }
}

#[test]
fn same_height_base_case_accepts_mixed_shapes_and_widths() {
    for seed in 100..108 {
        let (config, mmcs, proof, w, u, commits, target, source, challenger) =
            honest_run_with_batching(
                seed,
                0,
                0,
                Tamper::None,
                None,
                FreshMaskBatching::SameHeight,
                Some(mixed_groups()),
            );
        assert_eq!(proof.fresh_mask_commitments.len(), 2);
        assert_eq!(proof.fresh_mask_openings.len(), 2);
        assert_eq!(proof.carried_mask_openings.len(), 4);
        assert_eq!(proof.fresh_mask_openings[0].rows[0].len(), 6);
        assert!(
            verify_run(
                &config, &mmcs, &proof, &w, &u, &commits, target, &source, challenger
            )
            .is_ok()
        );
        let encoded = postcard::to_allocvec(&proof).unwrap();
        let decoded: BaseCaseZkProof<F, EF, MyMmcs> = postcard::from_bytes(&encoded).unwrap();
        assert_eq!(postcard::to_allocvec(&decoded).unwrap(), encoded);
    }
}

#[test]
fn same_height_base_case_rejects_distinct_malformed_payloads() {
    let (config, mmcs, proof, w, u, commits, target, source, challenger) = honest_run_with_batching(
        100,
        0,
        0,
        Tamper::None,
        None,
        FreshMaskBatching::SameHeight,
        Some(mixed_groups()),
    );
    let mut mutations = Vec::new();
    let mut changed = proof.clone();
    changed.fresh_mask_commitments[0] = proof.fresh_main_commitment.clone();
    mutations.push(changed);
    let mut changed = proof.clone();
    changed.fresh_mask_commitments.swap(0, 1);
    mutations.push(changed);
    let mut changed = proof.clone();
    changed.fresh_mask_openings[0].rows[0][0] += EF::ONE;
    mutations.push(changed);
    let mut changed = proof.clone();
    changed.carried_mask_openings[2].rows[0][0] += EF::ONE;
    mutations.push(changed);
    let mut changed = proof.clone();
    changed.fresh_mask_openings[0].rows[0].pop();
    mutations.push(changed);
    let mut changed = proof.clone();
    changed.carried_mask_openings[2].rows[0].pop();
    mutations.push(changed);
    let mut changed = proof.clone();
    changed.fresh_mask_openings[0].rows.pop();
    mutations.push(changed);
    let mut changed = proof.clone();
    changed.carried_mask_openings[2].rows.pop();
    mutations.push(changed);
    let mut changed = proof.clone();
    changed.fresh_mask_openings[0].proof.sibling_hashes[0][0] += F::ONE;
    mutations.push(changed);
    let mut changed = proof.clone();
    changed.carried_mask_openings[2].proof.sibling_hashes[0][0] += F::ONE;
    mutations.push(changed);
    let mut changed = proof.clone();
    changed.fresh_mask_openings.pop();
    mutations.push(changed);
    let mut changed = proof.clone();
    changed.carried_mask_openings.swap(0, 2);
    mutations.push(changed);
    let mut changed = proof.clone();
    changed.masked_claim += EF::ONE;
    mutations.push(changed);
    for (index, changed) in mutations.iter().enumerate() {
        assert!(
            verify_run(
                &config,
                &mmcs,
                changed,
                &w,
                &u,
                &commits,
                target,
                &source,
                challenger.clone()
            )
            .is_err(),
            "accepted mutation {index}"
        );
    }
    let mut changed_commits = commits.clone();
    changed_commits[2] = proof.fresh_mask_commitments[0].clone();
    assert!(
        verify_run(
            &config,
            &mmcs,
            &proof,
            &w,
            &u,
            &changed_commits,
            target,
            &source,
            challenger.clone()
        )
        .is_err()
    );
    let mut changed_config = config.clone();
    changed_config.mask_groups[2].width += 1;
    assert!(
        verify_run(
            &changed_config,
            &mmcs,
            &proof,
            &w,
            &u,
            &commits,
            target,
            &source,
            challenger.clone()
        )
        .is_err()
    );
    let mut changed_config = config.clone();
    changed_config.mask_groups.swap(1, 2);
    assert!(
        verify_run(
            &changed_config,
            &mmcs,
            &proof,
            &w,
            &u,
            &commits,
            target,
            &source,
            challenger.clone()
        )
        .is_err()
    );
    let mut changed_config = config.clone();
    changed_config.fresh_mask_batching = FreshMaskBatching::Separate;
    assert!(
        verify_run(
            &changed_config,
            &mmcs,
            &proof,
            &w,
            &u,
            &commits,
            target,
            &source,
            challenger
        )
        .is_err()
    );
}

#[test]
fn fresh_batch_roots_and_claim_bind_gamma() {
    let (_, _, proof, ..) = honest_run_with_batching(
        105,
        0,
        0,
        Tamper::None,
        None,
        FreshMaskBatching::SameHeight,
        Some(mixed_groups()),
    );
    let gamma = |proof: &BaseCaseZkProof<F, EF, MyMmcs>| -> EF {
        let (mut challenger, ..) = setup();
        challenger.observe(proof.fresh_main_commitment.clone());
        for commitment in &proof.fresh_mask_commitments {
            challenger.observe(commitment.clone());
        }
        challenger.observe_algebra_element(proof.masked_claim);
        challenger.sample_algebra_element()
    };
    let expected = gamma(&proof);
    for index in 0..proof.fresh_mask_commitments.len() {
        let mut changed = proof.clone();
        changed.fresh_mask_commitments[index] = proof.fresh_main_commitment.clone();
        assert_ne!(gamma(&changed), expected);
    }
    let mut changed = proof.clone();
    changed.fresh_mask_commitments.swap(0, 1);
    assert_ne!(gamma(&changed), expected);
    let mut changed = proof.clone();
    changed.masked_claim += EF::ONE;
    assert_ne!(gamma(&changed), expected);
}
