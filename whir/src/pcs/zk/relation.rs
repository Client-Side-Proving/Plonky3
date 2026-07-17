//! Public inputs for continuing HVZK-WHIR from an existing committed relation.

use alloc::vec::Vec;

use p3_commit::{ExtensionMmcs, Mmcs};
use p3_field::{ExtensionField, Field, TwoAdicField, dot_product};
use p3_matrix::dense::RowMajorMatrix;
use thiserror::Error;

use super::constraint::{SourceClaim, SourceTerm};
use super::mask::MaskGroupShape;

/// Merkle prover data for one extension-field mask group.
pub type MaskGroupProverData<F, EF, MT> =
    <ExtensionMmcs<F, EF, MT> as Mmcs<EF>>::ProverData<RowMajorMatrix<EF>>;

/// A committed-sumcheck relation carried into HVZK-WHIR.
///
/// The external mask groups supplied alongside this value complete the relation
///
/// ```text
///     <f, source> + sum_i <mask_i, covector_i> = target.
/// ```
#[derive(Debug, Clone)]
pub struct CommittedRelation<EF> {
    /// Symbolic constraints on the committed source message.
    pub source: SourceClaim<EF>,
    /// Public right-hand side of the joint relation.
    pub target: EF,
}

impl<EF: Field> CommittedRelation<EF> {
    /// A relation with no source constraints and target zero.
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            source: SourceClaim::new(),
            target: EF::ZERO,
        }
    }

    /// Checks that every symbolic source term addresses the committed arity.
    pub fn validate(&self, num_variables: usize) -> Result<(), RelationError> {
        for (constraint, source) in self.source.constraints.iter().enumerate() {
            let actual = match &source.term {
                SourceTerm::Eq { point } => point.num_variables(),
                SourceTerm::Pow { num_variables, .. } => *num_variables,
            };
            if actual != num_variables {
                return Err(RelationError::SourceArityMismatch {
                    constraint,
                    expected: num_variables,
                    actual,
                });
            }
        }
        Ok(())
    }
}

/// Prover-side state behind one already-committed mask group.
pub struct CommittedMaskGroupProverData<F, EF, MT>
where
    F: TwoAdicField,
    EF: ExtensionField<F>,
    MT: Mmcs<F>,
{
    /// Public code and group width.
    pub shape: MaskGroupShape,
    /// Public group commitment.
    pub commitment: MT::Commitment,
    /// Secret mask messages, one per group column.
    pub messages: Vec<Vec<EF>>,
    /// Secret encoding randomness, one per group column.
    pub randomness: Vec<Vec<EF>>,
    /// Dense relation covectors, one per group column.
    pub covectors: Vec<Vec<EF>>,
    /// Merkle prover data used by the final base case.
    pub prover_data: MaskGroupProverData<F, EF, MT>,
}

impl<F, EF, MT> CommittedMaskGroupProverData<F, EF, MT>
where
    F: TwoAdicField,
    EF: ExtensionField<F>,
    MT: Mmcs<F>,
{
    /// Checks the group shape and every message/covector dimension.
    pub fn validate(&self) -> Result<(), RelationError> {
        validate_group_dimensions(
            self.shape,
            &self.messages,
            &self.randomness,
            &self.covectors,
        )
    }

    /// Evaluates this group's contribution to the joint relation.
    pub fn claim(&self) -> EF {
        self.messages
            .iter()
            .zip(&self.covectors)
            .map(|(message, covector)| {
                dot_product::<EF, _, _>(message.iter().copied(), covector.iter().copied())
            })
            .sum()
    }
}

/// Verifier-side description of one already-observed mask commitment.
#[derive(Debug, Clone)]
pub struct CommittedMaskGroup<EF, Commitment> {
    /// Public code and group width.
    pub shape: MaskGroupShape,
    /// Public group commitment.
    pub commitment: Commitment,
    /// Dense relation covectors, one per group column.
    pub covectors: Vec<Vec<EF>>,
}

impl<EF: Field, Commitment> CommittedMaskGroup<EF, Commitment> {
    /// Checks the public group shape and covector dimensions.
    pub fn validate(&self) -> Result<(), RelationError> {
        if self.covectors.len() != self.shape.width {
            return Err(RelationError::GroupWidthMismatch {
                expected: self.shape.width,
                actual: self.covectors.len(),
            });
        }
        for (mask, covector) in self.covectors.iter().enumerate() {
            if covector.len() != self.shape.shape.message_len {
                return Err(RelationError::CovectorLengthMismatch {
                    mask,
                    expected: self.shape.shape.message_len,
                    actual: covector.len(),
                });
            }
        }
        Ok(())
    }
}

fn validate_group_dimensions<EF: Field>(
    shape: MaskGroupShape,
    messages: &[Vec<EF>],
    randomness: &[Vec<EF>],
    covectors: &[Vec<EF>],
) -> Result<(), RelationError> {
    for actual in [messages.len(), randomness.len(), covectors.len()] {
        if actual != shape.width {
            return Err(RelationError::GroupWidthMismatch {
                expected: shape.width,
                actual,
            });
        }
    }
    for (mask, message) in messages.iter().enumerate() {
        if message.len() != shape.shape.message_len {
            return Err(RelationError::MessageLengthMismatch {
                mask,
                expected: shape.shape.message_len,
                actual: message.len(),
            });
        }
    }
    for (mask, randomness) in randomness.iter().enumerate() {
        if randomness.len() != shape.shape.randomness_len {
            return Err(RelationError::RandomnessLengthMismatch {
                mask,
                expected: shape.shape.randomness_len,
                actual: randomness.len(),
            });
        }
    }
    for (mask, covector) in covectors.iter().enumerate() {
        if covector.len() != shape.shape.message_len {
            return Err(RelationError::CovectorLengthMismatch {
                mask,
                expected: shape.shape.message_len,
                actual: covector.len(),
            });
        }
    }
    Ok(())
}

/// Invalid committed-relation input.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum RelationError {
    /// A symbolic source term addresses the wrong source arity.
    #[error("source constraint {constraint}: expected arity {expected}, got {actual}")]
    SourceArityMismatch {
        /// Constraint index.
        constraint: usize,
        /// Committed source arity.
        expected: usize,
        /// Supplied source arity.
        actual: usize,
    },
    /// A group does not contain its declared number of masks.
    #[error("mask group width mismatch: expected {expected}, got {actual}")]
    GroupWidthMismatch {
        /// Declared width.
        expected: usize,
        /// Supplied width.
        actual: usize,
    },
    /// A mask message has the wrong length.
    #[error("mask {mask}: expected message length {expected}, got {actual}")]
    MessageLengthMismatch {
        /// Mask index in its group.
        mask: usize,
        /// Declared message length.
        expected: usize,
        /// Supplied message length.
        actual: usize,
    },
    /// A mask randomness vector has the wrong length.
    #[error("mask {mask}: expected randomness length {expected}, got {actual}")]
    RandomnessLengthMismatch {
        /// Mask index in its group.
        mask: usize,
        /// Declared randomness length.
        expected: usize,
        /// Supplied randomness length.
        actual: usize,
    },
    /// A mask covector has the wrong length.
    #[error("mask {mask}: expected covector length {expected}, got {actual}")]
    CovectorLengthMismatch {
        /// Mask index in its group.
        mask: usize,
        /// Declared covector length.
        expected: usize,
        /// Supplied covector length.
        actual: usize,
    },
}
