/*=========================================================================
 *
 *  Copyright UMC Utrecht and contributors
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
#ifndef _itkMultiPWMutualInformationImageToImageMetric_HXX__
#define _itkMultiPWMutualInformationImageToImageMetric_HXX__

#include "itkMultiPWMutualInformationImageToImageMetric.h"

#include "itkImageLinearConstIteratorWithIndex.h"
#include "itkImageScanlineConstIterator.h"
#include "vnl/vnl_math.h"
#include "itkMatrix.h"
#include "vnl/vnl_inverse.h"
#include "vnl/vnl_det.h"

#ifdef ELASTIX_USE_OPENMP
#include <omp.h>
#endif

namespace itk
{
/**
 * ********************* Constructor ******************************
 */

template< class TFixedImage, class TMovingImage >
MultiPWMutualInformationImageToImageMetric< TFixedImage, TMovingImage >
::MultiPWMutualInformationImageToImageMetric()
{
} // end constructor


/**
 * ********************* InitializeHistograms ******************************
 */

template< class TFixedImage, class TMovingImage >
void
MultiPWMutualInformationImageToImageMetric< TFixedImage, TMovingImage >
::InitializeHistograms( void )
{
  /** Call Superclass implementation. */
  this->Superclass::InitializeHistograms();
} // end InitializeHistograms()


/**
 * ************************** GetValue **************************
 */

template< class TFixedImage, class TMovingImage >
typename MultiPWMutualInformationImageToImageMetric< TFixedImage, TMovingImage >
::MeasureType
MultiPWMutualInformationImageToImageMetric< TFixedImage, TMovingImage >
::GetValue( const ParametersType & parameters ) const
{
  /** Get the number of multiple images in the cluster */
  if (this->GetNumberOfFixedImages() != this->GetNumberOfMovingImages())
    itkExceptionMacro(<< "MultiNormalizedCorrelationImageToImageMetric requires the same number of fixed and moving images");

  const unsigned int clusterSize = this->GetNumberOfFixedImages();
  /** Loop over all the multiple images in the cluster */
  double MI = 0.0;
  for (unsigned int pos = 0; pos < clusterSize; ++pos)
  {
    /** Construct the JointPDF and Alpha. */
    this->ComputePDFs(parameters, pos);

    /** Normalize the pdfs: p = alpha h. */
    this->NormalizeJointPDF(this->m_JointPDFVector[pos], this->m_AlphaVector[pos]);

    /** Compute the fixed and moving marginal pdfs, by summing over the joint pdf. */
    this->ComputeMarginalPDF(this->m_JointPDFVector[pos], this->m_FixedImageMarginalPDFVector[pos], 0, pos);
    this->ComputeMarginalPDF(this->m_JointPDFVector[pos], this->m_MovingImageMarginalPDFVector[pos], 1, pos);

    /** Compute the metric by double summation over histogram. */

    /** Setup iterators */
    typedef ImageLinearConstIteratorWithIndex< JointPDFType > JointPDFIteratorType;
    typedef typename MarginalPDFType::const_iterator          MarginalPDFIteratorType;

    JointPDFIteratorType jointPDFit(
      this->m_JointPDFVector[pos], this->m_JointPDFVector[pos]->GetLargestPossibleRegion());
    jointPDFit.SetDirection(0);
    jointPDFit.GoToBegin();
    MarginalPDFIteratorType       fixedPDFit = this->m_FixedImageMarginalPDFVector[pos].begin();
    const MarginalPDFIteratorType fixedPDFend = this->m_FixedImageMarginalPDFVector[pos].end();
    MarginalPDFIteratorType       movingPDFit = this->m_MovingImageMarginalPDFVector[pos].begin();
    const MarginalPDFIteratorType movingPDFend = this->m_MovingImageMarginalPDFVector[pos].end();

    /** Loop over histogram. */
    MI = 0.0;
    while (fixedPDFit != fixedPDFend)
    {
      const double fixedImagePDFValue = *fixedPDFit;
      movingPDFit = this->m_MovingImageMarginalPDFVector[pos].begin();
      while (movingPDFit != movingPDFend)
      {
        const double movingImagePDFValue = *movingPDFit;
        const double fixPDFmovPDF = fixedImagePDFValue * movingImagePDFValue;
        const double jointPDFValue = jointPDFit.Get();

        /** Check for non-zero bin contribution. */
        if (jointPDFValue > 1e-16 && fixPDFmovPDF > 1e-16)
        {
          MI += jointPDFValue * std::log(jointPDFValue / fixPDFmovPDF);
        }
        ++movingPDFit;
        ++jointPDFit;
      }  // end while-loop over moving index

      ++fixedPDFit;
      jointPDFit.NextLine();

    } // end while-loop over fixed index
  }
  return static_cast< MeasureType >( -1.0 * MI );

} // end GetValue()


/**
 * ******************** GetValueAndAnalyticDerivative *******************
 */
template< class TFixedImage, class TMovingImage >
void
MultiPWMutualInformationImageToImageMetric< TFixedImage, TMovingImage >
::GetValueAndAnalyticDerivative(
  const ParametersType & parameters,
  MeasureType & value,
  DerivativeType & derivative ) const
{
  /** Get the number of multiple images in the cluster */
  if (this->GetNumberOfFixedImages() != this->GetNumberOfMovingImages())
    itkExceptionMacro(<< "MultiNormalizedCorrelationImageToImageMetric requires the same number of fixed and moving images");

  const unsigned int clusterSize = this->GetNumberOfFixedImages();
  /** Loop over all the multiple images in the cluster */
  /** Initialize some variables. */
  value = NumericTraits< MeasureType >::Zero;
  derivative = DerivativeType(this->GetNumberOfParameters());
  derivative.Fill(NumericTraits< double >::ZeroValue());
  double MI = 0.0;
  for (unsigned int pos = 0; pos < clusterSize; ++pos)
  {
      /** Construct the JointPDF, JointPDFDerivatives, Alpha and its derivatives. */
      this->ComputePDFsAndPDFDerivatives(parameters, pos);

      /** Normalize the pdfs: p = alpha h. */
      this->NormalizeJointPDF(this->m_JointPDFVector[pos], this->m_AlphaVector[pos]);
      /** Compute the fixed and moving marginal pdf by summing over the histogram. */
      this->ComputeMarginalPDF(this->m_JointPDFVector[pos], this->m_FixedImageMarginalPDFVector[pos], 0, pos);
      this->ComputeMarginalPDF(this->m_JointPDFVector[pos], this->m_MovingImageMarginalPDFVector[pos], 1, pos);

      /** Compute the metric and derivatives by double summation over histogram. */
      /** Setup iterators .*/
      typedef ImageLinearConstIteratorWithIndex<
        JointPDFType >                                 JointPDFIteratorType;
      typedef ImageLinearConstIteratorWithIndex<
        JointPDFDerivativesType >                       JointPDFDerivativesIteratorType;
      typedef typename MarginalPDFType::const_iterator MarginalPDFIteratorType;
      typedef typename DerivativeType::iterator        DerivativeIteratorType;

      JointPDFIteratorType jointPDFit(
        this->m_JointPDFVector[pos], this->m_JointPDFVector[pos]->GetLargestPossibleRegion());
      jointPDFit.SetDirection(0);
      jointPDFit.GoToBegin();
      JointPDFDerivativesIteratorType jointPDFDerivativesit(
        this->m_JointPDFDerivativesVector[pos], this->m_JointPDFDerivativesVector[pos]->GetLargestPossibleRegion());
      jointPDFDerivativesit.SetDirection(0);
      jointPDFDerivativesit.GoToBegin();
      MarginalPDFIteratorType       fixedPDFit = this->m_FixedImageMarginalPDFVector[pos].begin();
      const MarginalPDFIteratorType fixedPDFend = this->m_FixedImageMarginalPDFVector[pos].end();
      MarginalPDFIteratorType       movingPDFit = this->m_MovingImageMarginalPDFVector[pos].begin();
      const MarginalPDFIteratorType movingPDFend = this->m_MovingImageMarginalPDFVector[pos].end();
      DerivativeIteratorType        derivit = derivative.begin();
      const DerivativeIteratorType  derivbegin = derivative.begin();
      const DerivativeIteratorType  derivend = derivative.end();
      /** Loop over the joint histogram. */
      while (fixedPDFit != fixedPDFend)
      {
        const double fixedImagePDFValue = *fixedPDFit;
        movingPDFit = this->m_MovingImageMarginalPDFVector[pos].begin();
        while (movingPDFit != movingPDFend)
        {
          const double movingImagePDFValue = *movingPDFit;
          const double fixPDFmovPDF = fixedImagePDFValue * movingImagePDFValue;
          const double jointPDFValue = jointPDFit.Get();

          /** Check for non-zero bin contribution. */
          if (jointPDFValue > 1e-16 && fixPDFmovPDF > 1e-16)
          {
            derivit = derivbegin;
            const double pRatio = std::log(jointPDFValue / fixPDFmovPDF);
            const double pRatioAlpha = this->m_AlphaVector[pos] * pRatio;
            MI += jointPDFValue * pRatio;
            while (derivit != derivend)
            {
              /**  Ref: eq 23 of Thevenaz & Unser paper [3]. */
              (*derivit) -= jointPDFDerivativesit.Get() * pRatioAlpha;
              ++derivit;
              ++jointPDFDerivativesit;
            } // end while-loop over parameters
          } // end if-block to check non-zero bin contribution

          ++movingPDFit;
          ++jointPDFit;
          jointPDFDerivativesit.NextLine();

        }  // end while-loop over moving index
        ++fixedPDFit;
        jointPDFit.NextLine();
      }  // end while-loop over fixed index
    value += static_cast<MeasureType>(-1.0 * MI);

  } // end for-loop over image cluster
  value = static_cast<MeasureType>(value/clusterSize);
  derivative /= clusterSize;
} // end GetValueAndAnalyticDerivative()



#ifdef BENCHMARK  /* Just for TESTING analytic derivative vs. numeric (finite difference) */
/**
 * ******************** GetValueAndFiniteDifferenceDerivative *******************
 */
template< class TFixedImage, class TMovingImage >
void
MultiPWMutualInformationImageToImageMetric< TFixedImage, TMovingImage >
::GetValueAndFiniteDifferenceDerivative(
  const ParametersType& parameters,
  MeasureType& value,
  DerivativeType& derivative) const
{
  /** Get the number of multiple images in the cluster */
  if (this->GetNumberOfFixedImages() != this->GetNumberOfMovingImages())
    itkExceptionMacro(<< "MultiNormalizedCorrelationImageToImageMetric requires the same number of fixed and moving images");

  const unsigned int clusterSize = this->GetNumberOfFixedImages();

  const double delta2 = -1.0 / (this->GetFiniteDifferencePerturbation() * 2.0);

  /** Loop over all the multiple images in the cluster */
  /** Initialize some variables. */
  value = NumericTraits< MeasureType >::Zero;
  derivative = DerivativeType(this->GetNumberOfParameters());
  derivative.Fill(NumericTraits< double >::ZeroValue());
  double MI = 0.0;
  for (unsigned int pos = 0; pos < clusterSize; ++pos)
  {
    /** Construct the JointPDF, JointPDFDerivatives, Alpha and its derivatives. */
    this->ComputePDFsAndIncrementalPDFs(parameters, pos);

    /** Compute the fixed and moving marginal pdf by summing over the histogram. */
    this->ComputeMarginalPDF(this->m_JointPDFVector[pos], this->m_FixedImageMarginalPDFVector[pos], 0, pos);
    this->ComputeMarginalPDF(this->m_JointPDFVector[pos], this->m_MovingImageMarginalPDFVector[pos], 1, pos);

    /** Compute the fixed and moving incremental marginal pdfs by summing over the
     * incremental histogram. Do it for Right and Left.
     */
    this->ComputeIncrementalMarginalPDFs(this->m_IncrementalJointPDFRightVector[pos],
      this->m_FixedIncrementalMarginalPDFRightVector[pos], this->m_MovingIncrementalMarginalPDFRightVector[pos]);
    this->ComputeIncrementalMarginalPDFs(this->m_IncrementalJointPDFLeftVector[pos],
      this->m_FixedIncrementalMarginalPDFLeftVector[pos], this->m_MovingIncrementalMarginalPDFLeftVector[pos]);

    /** Compute the metric and derivatives by double summation over histogram. */

    /** Setup iterators */
    typedef ImageLinearConstIteratorWithIndex<
      JointPDFType >                                 JointPDFIteratorType;
    typedef ImageLinearConstIteratorWithIndex<
      JointPDFDerivativesType >                       IncrementalJointPDFIteratorType;
    typedef typename MarginalPDFType::const_iterator MarginalPDFIteratorType;
    typedef ImageLinearConstIteratorWithIndex<
      IncrementalMarginalPDFType >                   IncrementalMarginalPDFIteratorType;
    typedef typename DerivativeType::iterator       DerivativeIteratorType;
    typedef typename DerivativeType::const_iterator DerivativeConstIteratorType;

    JointPDFIteratorType jointPDFit(
      this->m_JointPDFVector[pos], this->m_JointPDFVector[pos]->GetLargestPossibleRegion());
    jointPDFit.GoToBegin();

    IncrementalJointPDFIteratorType jointIncPDFRightit(this->m_IncrementalJointPDFRightVector[pos],
      this->m_IncrementalJointPDFRightVector[pos]->GetLargestPossibleRegion());
    IncrementalJointPDFIteratorType jointIncPDFLeftit(this->m_IncrementalJointPDFLeftVector[pos],
      this->m_IncrementalJointPDFLeftVector[pos]->GetLargestPossibleRegion());
    jointIncPDFRightit.GoToBegin();
    jointIncPDFLeftit.GoToBegin();

    MarginalPDFIteratorType       fixedPDFit = this->m_FixedImageMarginalPDFVector[pos].begin();
    const MarginalPDFIteratorType fixedPDFend = this->m_FixedImageMarginalPDFVector[pos].end();
    MarginalPDFIteratorType       movingPDFit = this->m_MovingImageMarginalPDFVector[pos].begin();
    const MarginalPDFIteratorType movingPDFend = this->m_MovingImageMarginalPDFVector[pos].end();

    IncrementalMarginalPDFIteratorType fixedIncPDFRightit(
      this->m_FixedIncrementalMarginalPDFRightVector[pos],
      this->m_FixedIncrementalMarginalPDFRightVector[pos]->GetLargestPossibleRegion());
    IncrementalMarginalPDFIteratorType movingIncPDFRightit(
      this->m_MovingIncrementalMarginalPDFRightVector[pos],
      this->m_MovingIncrementalMarginalPDFRightVector[pos]->GetLargestPossibleRegion());
    IncrementalMarginalPDFIteratorType fixedIncPDFLeftit(
      this->m_FixedIncrementalMarginalPDFLeftVector[pos],
      this->m_FixedIncrementalMarginalPDFLeftVector[pos]->GetLargestPossibleRegion());
    IncrementalMarginalPDFIteratorType movingIncPDFLeftit(
      this->m_MovingIncrementalMarginalPDFLeftVector[pos],
      this->m_MovingIncrementalMarginalPDFLeftVector[pos]->GetLargestPossibleRegion());
    fixedIncPDFRightit.GoToBegin();
    movingIncPDFRightit.GoToBegin();
    fixedIncPDFLeftit.GoToBegin();
    movingIncPDFLeftit.GoToBegin();

    DerivativeIteratorType       derivit = derivative.begin();
    const DerivativeIteratorType derivbegin = derivative.begin();
    const DerivativeIteratorType derivend = derivative.end();

    DerivativeConstIteratorType       perturbedAlphaRightit = this->m_PerturbedAlphaRightVector[pos].begin();
    const DerivativeConstIteratorType perturbedAlphaRightbegin = this->m_PerturbedAlphaRightVector[pos].begin();
    DerivativeConstIteratorType       perturbedAlphaLeftit = this->m_PerturbedAlphaLeftVector[pos].begin();
    const DerivativeConstIteratorType perturbedAlphaLeftbegin = this->m_PerturbedAlphaLeftVector[pos].begin();

    while (fixedPDFit != fixedPDFend)
    {
      const double fixedPDFValue = *fixedPDFit;

      while (movingPDFit != movingPDFend)
      {
        const double movingPDFValue = *movingPDFit;
        const double jointPDFValue = jointPDFit.Get();
        const double fixPDFmovPDFAlpha
          = fixedPDFValue * movingPDFValue * this->m_AlphaVector[pos];

        /** Check for non-zero bin contribution and update the mutual information value. */
        if (jointPDFValue > 1e-16 && fixPDFmovPDFAlpha > 1e-16)
        {
          MI += this->m_AlphaVector[pos] * jointPDFValue * std::log(jointPDFValue / fixPDFmovPDFAlpha);
        }

        /** Update the derivative. */
        derivit = derivbegin;
        perturbedAlphaRightit = perturbedAlphaRightbegin;
        perturbedAlphaLeftit = perturbedAlphaLeftbegin;
        while (derivit != derivend)
        {
          /** Initialize. */
          double contrib = 0.0;

          /** For clarity, get some values and give them a name.
           * \todo Does this cost a lot of computation time?
           */
          const double jointIncPDFRightValue = jointIncPDFRightit.Get();
          const double fixedIncPDFRightValue = fixedIncPDFRightit.Get();
          const double movingIncPDFRightValue = movingIncPDFRightit.Get();
          const double perturbedAlphaRightValue = *perturbedAlphaRightit;

          /** Compute the contribution of the Right-perturbation to the derivative. */
          const double perturbedJointPDFRightValue = jointIncPDFRightValue + jointPDFValue;
          const double perturbedFixedPDFRightValue = fixedPDFValue + fixedIncPDFRightValue;
          const double perturbedMovingPDFRightValue = movingPDFValue + movingIncPDFRightValue;
          const double perturbedfixPDFmovPDFAlphaRight
            = perturbedFixedPDFRightValue * perturbedMovingPDFRightValue * perturbedAlphaRightValue;

          if (perturbedJointPDFRightValue > 1e-16 && perturbedfixPDFmovPDFAlphaRight > 1e-16)
          {
            contrib += perturbedAlphaRightValue * perturbedJointPDFRightValue
              * std::log(perturbedJointPDFRightValue / perturbedfixPDFmovPDFAlphaRight);
          }

          /** For clarity, get some values and give them a name. */
          const double jointIncPDFLeftValue = jointIncPDFLeftit.Get();
          const double fixedIncPDFLeftValue = fixedIncPDFLeftit.Get();
          const double movingIncPDFLeftValue = movingIncPDFLeftit.Get();
          const double perturbedAlphaLeftValue = *perturbedAlphaLeftit;

          /** Compute the contribution of the Left-perturbation to the derivative. */
          const double perturbedJointPDFLeftValue = jointIncPDFLeftValue + jointPDFValue;
          const double perturbedFixedPDFLeftValue = fixedPDFValue + fixedIncPDFLeftValue;
          const double perturbedMovingPDFLeftValue = movingPDFValue + movingIncPDFLeftValue;
          const double perturbedfixPDFmovPDFAlphaLeft
            = perturbedFixedPDFLeftValue * perturbedMovingPDFLeftValue * perturbedAlphaLeftValue;

          if (perturbedJointPDFLeftValue > 1e-16 && perturbedfixPDFmovPDFAlphaLeft > 1e-16)
          {
            contrib -= perturbedAlphaLeftValue * perturbedJointPDFLeftValue
              * std::log(perturbedJointPDFLeftValue / perturbedfixPDFmovPDFAlphaLeft);
          }

          /** Update the derivative component and divide the derivative by -delta*2. */
          (*derivit) += contrib*delta2;
   

          /** Move the iterators to the next parameter. */
          ++derivit;
          ++perturbedAlphaRightit;
          ++perturbedAlphaLeftit;
          ++jointIncPDFRightit;
          ++jointIncPDFLeftit;
          ++fixedIncPDFRightit;
          ++movingIncPDFRightit;
          ++fixedIncPDFLeftit;
          ++movingIncPDFLeftit;
        }  // end while-loop over parameters

        ++jointPDFit;                         // next moving bin
        ++movingPDFit;                        // next moving bin
        jointIncPDFRightit.NextLine();        // next moving bin
        jointIncPDFLeftit.NextLine();         // next moving bin
        fixedIncPDFRightit.GoToBeginOfLine(); // same fixed bin
        fixedIncPDFLeftit.GoToBeginOfLine();  // same fixed bin
        movingIncPDFRightit.NextLine();       // next moving bin
        movingIncPDFLeftit.NextLine();        // next moving bin

      }  // end while-loop over moving index

      jointPDFit.NextLine();                                // next fixed bin
      ++fixedPDFit;                                         // next fixed bin
      movingPDFit = this->m_MovingImageMarginalPDFVector[pos].begin(); // first moving bin
      fixedIncPDFRightit.NextLine();                        // next fixed bin
      fixedIncPDFLeftit.NextLine();                         // next fixed bin
      movingIncPDFRightit.GoToBegin();                      // first moving bin
      movingIncPDFLeftit.GoToBegin();                       // first moving bin
  }  // end while-loop over fixed index
  } // end for-loop over image cluster
  value = static_cast<MeasureType>(-1.0 * MI/clusterSize);
  derivative /= clusterSize;
} // end GetValueAndFiniteDifferenceDerivative

#endif


} // end namespace itk

#endif // end #ifndef _itkMultiPWMutualInformationImageToImageMetric_HXX__
