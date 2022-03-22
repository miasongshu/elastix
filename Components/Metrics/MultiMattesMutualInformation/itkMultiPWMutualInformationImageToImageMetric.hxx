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
    this->NormalizeJointPDF(this->m_JointPDF, this->m_Alpha);

    /** Compute the fixed and moving marginal pdfs, by summing over the joint pdf. */
    this->ComputeMarginalPDF(this->m_JointPDF, this->m_FixedImageMarginalPDF, 0);
    this->ComputeMarginalPDF(this->m_JointPDF, this->m_MovingImageMarginalPDF, 1);

    /** Compute the metric by double summation over histogram. */

    /** Setup iterators */
    typedef ImageLinearConstIteratorWithIndex< JointPDFType > JointPDFIteratorType;
    typedef typename MarginalPDFType::const_iterator          MarginalPDFIteratorType;

    JointPDFIteratorType jointPDFit(
      this->m_JointPDF, this->m_JointPDF->GetLargestPossibleRegion());
    jointPDFit.SetDirection(0);
    jointPDFit.GoToBegin();
    MarginalPDFIteratorType       fixedPDFit = this->m_FixedImageMarginalPDF.begin();
    const MarginalPDFIteratorType fixedPDFend = this->m_FixedImageMarginalPDF.end();
    MarginalPDFIteratorType       movingPDFit = this->m_MovingImageMarginalPDF.begin();
    const MarginalPDFIteratorType movingPDFend = this->m_MovingImageMarginalPDF.end();

    /** Loop over histogram. */
    while (fixedPDFit != fixedPDFend)
    {
      const double fixedImagePDFValue = *fixedPDFit;
      movingPDFit = this->m_MovingImageMarginalPDF.begin();
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
  double MI = 0.0;
  for (unsigned int pos = 0; pos < clusterSize; ++pos)
  {
    /** Initialize some variables. */
    value = NumericTraits< MeasureType >::Zero;
    derivative = DerivativeType(this->GetNumberOfParameters());
    derivative.Fill(NumericTraits< double >::ZeroValue());

    {
      /** Construct the JointPDF, JointPDFDerivatives, Alpha and its derivatives. */
      this->ComputePDFsAndPDFDerivatives(parameters, pos);
      /** Normalize the pdfs: p = alpha h. */
      this->NormalizeJointPDF(this->m_JointPDF, this->m_Alpha);
      /** Compute the fixed and moving marginal pdf by summing over the histogram. */
      this->ComputeMarginalPDF(this->m_JointPDF, this->m_FixedImageMarginalPDF, 0);
      this->ComputeMarginalPDF(this->m_JointPDF, this->m_MovingImageMarginalPDF, 1);
      /** Compute the metric and derivatives by double summation over histogram. */

      /** Setup iterators .*/
      typedef ImageLinearConstIteratorWithIndex<
        JointPDFType >                                 JointPDFIteratorType;
      typedef ImageLinearConstIteratorWithIndex<
        JointPDFDerivativesType >                       JointPDFDerivativesIteratorType;
      typedef typename MarginalPDFType::const_iterator MarginalPDFIteratorType;
      typedef typename DerivativeType::iterator        DerivativeIteratorType;

      JointPDFIteratorType jointPDFit(
        this->m_JointPDF, this->m_JointPDF->GetLargestPossibleRegion());
      jointPDFit.SetDirection(0);
      jointPDFit.GoToBegin();
      JointPDFDerivativesIteratorType jointPDFDerivativesit(
        this->m_JointPDFDerivatives, this->m_JointPDFDerivatives->GetLargestPossibleRegion());
      jointPDFDerivativesit.SetDirection(0);
      jointPDFDerivativesit.GoToBegin();
      MarginalPDFIteratorType       fixedPDFit = this->m_FixedImageMarginalPDF.begin();
      const MarginalPDFIteratorType fixedPDFend = this->m_FixedImageMarginalPDF.end();
      MarginalPDFIteratorType       movingPDFit = this->m_MovingImageMarginalPDF.begin();
      const MarginalPDFIteratorType movingPDFend = this->m_MovingImageMarginalPDF.end();
      DerivativeIteratorType        derivit = derivative.begin();
      const DerivativeIteratorType  derivbegin = derivative.begin();
      const DerivativeIteratorType  derivend = derivative.end();
      /** Loop over the joint histogram. */
      while (fixedPDFit != fixedPDFend)
      {
        const double fixedImagePDFValue = *fixedPDFit;
        movingPDFit = this->m_MovingImageMarginalPDF.begin();
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
            const double pRatioAlpha = this->m_Alpha * pRatio;
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
    }
    value += static_cast<MeasureType>(-1.0 * MI);
  }
} // end GetValueAndAnalyticDerivative()



} // end namespace itk

#endif // end #ifndef _itkMultiPWMutualInformationImageToImageMetric_HXX__
