//! Mask-code geometry for the HVZK-WHIR pipeline.
//!
//! Code shapes and committed mask groups, shared by the configuration, the
//! verifier replay, and the base case.

use alloc::vec::Vec;

use p3_dft::Radix2Dit;
use p3_field::TwoAdicField;
use p3_zk_codes::ReedSolomonZkEncoding;
use rand::distr::{Distribution, StandardUniform};
use serde::{Deserialize, Serialize};

/// Shape of one small mask code: a Reed-Solomon ZK encoding over `EF`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MaskCodeShape {
    /// Message length.
    pub message_len: usize,
    /// ZK randomness coefficients.
    pub randomness_len: usize,
    /// Codeword length (power of two).
    pub domain_size: usize,
}

impl MaskCodeShape {
    /// Derives the smallest power-of-two domain for the requested rate.
    #[must_use]
    pub const fn new(message_len: usize, randomness_len: usize, log_inv_rate: usize) -> Self {
        let domain_size = ((message_len + randomness_len).next_power_of_two()) << log_inv_rate;
        Self {
            message_len,
            randomness_len,
            domain_size,
        }
    }

    /// Instantiates the encoding over the extension field.
    ///
    /// Mask codewords are tiny, so a fresh radix-2 DIT per call is free.
    #[must_use]
    pub fn encoding<EF>(&self) -> ReedSolomonZkEncoding<EF, Radix2Dit<EF>>
    where
        EF: TwoAdicField,
        StandardUniform: Distribution<EF>,
    {
        ReedSolomonZkEncoding::new(
            self.randomness_len,
            self.message_len,
            self.domain_size,
            Radix2Dit::default(),
        )
    }
}

/// One committed batch of same-code mask oracles.
///
/// - Masks committed together share an evaluation domain.
/// - They stack into one matrix.
/// - A single commitment and one Merkle path per opened position cover the
///   whole group.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MaskGroupShape {
    /// Code shared by every mask in the group.
    pub shape: MaskCodeShape,
    /// Number of stacked masks.
    pub width: usize,
}

/// Fresh-mask commitment grouping at the terminal hiding base case.
///
/// This changes the proof schema and transcript, while preserving every
/// carried mask's code, independently sampled blind, and query count.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum FreshMaskBatching {
    /// Commit and query each fresh group separately.
    #[default]
    Separate,
    /// Concatenate fresh groups with equal domain sizes into one matrix.
    SameHeight,
}

/// Deterministically derived fresh-mask commitment geometry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FreshMaskBatch {
    /// Height shared by every member's encoded matrix.
    pub domain_size: usize,
    /// Total number of extension-field columns in the combined matrix.
    pub width: usize,
    /// Carried-group indices, in chronological order.
    pub group_indices: Vec<usize>,
}

impl FreshMaskBatching {
    /// Derives batches in order of first carried-group occurrence.
    ///
    /// Members and their columns remain in chronological order. Group
    /// mappings are derived from validated configuration, never from a proof.
    #[must_use]
    pub fn batches(self, groups: &[MaskGroupShape]) -> Vec<FreshMaskBatch> {
        let mut batches: Vec<FreshMaskBatch> = Vec::new();
        for (group_index, group) in groups.iter().enumerate() {
            let existing = if self == Self::SameHeight {
                batches
                    .iter_mut()
                    .find(|batch| batch.domain_size == group.shape.domain_size)
            } else {
                None
            };
            if let Some(batch) = existing {
                batch.width += group.width;
                batch.group_indices.push(group_index);
            } else {
                batches.push(FreshMaskBatch {
                    domain_size: group.shape.domain_size,
                    width: group.width,
                    group_indices: alloc::vec![group_index],
                });
            }
        }
        batches
    }
}
