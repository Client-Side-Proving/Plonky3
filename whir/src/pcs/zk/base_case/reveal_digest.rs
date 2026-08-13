use alloc::vec::Vec;

use p3_commit::{ExtensionMmcs, Mmcs};
use p3_field::{ExtensionField, Field};
use p3_matrix::dense::DenseMatrix;

use crate::pcs::zk::proof::BlindedMask;

const EXTENSION_ELEMENTS_PER_LEAF: usize = 2;

/// Commits to the ordered base-case reveal block for transcript absorption.
pub(super) fn reveal_digest<F, EF, MT>(
    extension_mmcs: &ExtensionMmcs<F, EF, MT>,
    blinded_message: &[EF],
    blinded_randomness: &[EF],
    blinded_masks: &[BlindedMask<EF>],
) -> MT::Commitment
where
    F: Field,
    EF: ExtensionField<F>,
    MT: Mmcs<F>,
{
    let mask_len = blinded_masks
        .iter()
        .map(|mask| mask.message.len() + mask.randomness.len())
        .sum::<usize>();
    let mut reveals =
        Vec::with_capacity(blinded_message.len() + blinded_randomness.len() + mask_len + 1);
    reveals.extend_from_slice(blinded_message);
    reveals.extend_from_slice(blinded_randomness);
    for mask in blinded_masks {
        reveals.extend_from_slice(&mask.message);
        reveals.extend_from_slice(&mask.randomness);
    }
    if !reveals.len().is_multiple_of(EXTENSION_ELEMENTS_PER_LEAF) {
        reveals.push(EF::ZERO);
    }

    extension_mmcs
        .commit_matrix(DenseMatrix::new(reveals, EXTENSION_ELEMENTS_PER_LEAF))
        .0
}
