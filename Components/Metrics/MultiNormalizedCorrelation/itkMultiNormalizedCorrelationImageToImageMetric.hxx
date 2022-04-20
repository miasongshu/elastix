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
#ifndef _itkMultiNormalizedCorrelationImageToImageMetric_hxx
#define _itkMultiNormalizedCorrelationImageToImageMetric_hxx

#include "itkMultiNormalizedCorrelationImageToImageMetric.h"

#ifdef ELASTIX_USE_OPENMP
#include <omp.h>
#endif

namespace itk
{

/**
 * ******************* Constructor *******************
 */

template< class TFixedImage, class TMovingImage >
MultiNormalizedCorrelationImageToImageMetric< TFixedImage, TMovingImage >
::MultiNormalizedCorrelationImageToImageMetric()
{
  this->m_SubtractMean = true;

  this->SetUseImageSampler( true );
  this->SetUseFixedImageLimiter( false );
  this->SetUseMovingImageLimiter( false );

} // end Constructor

/**
 * ******************* PrintSelf *******************
 */

template< class TFixedImage, class TMovingImage >
void
MultiNormalizedCorrelationImageToImageMetric< TFixedImage, TMovingImage >
::PrintSelf( std::ostream & os, Indent indent ) const
{
  Superclass::PrintSelf( os, indent );
  os << indent << "SubtractMean: " << this->m_SubtractMean << std::endl;

} // end PrintSelf()


/**
 * *************** UpdateDerivativeTerms ***************************
 */

template< class TFixedImage, class TMovingImage >
void
MultiNormalizedCorrelationImageToImageMetric< TFixedImage, TMovingImage >
::UpdateDerivativeTerms(
  const RealType & fixedImageValue,
  const RealType & movingImageValue,
  const DerivativeType & imageJacobian,
  const NonZeroJacobianIndicesType & nzji,
  DerivativeType & derivativeF,
  DerivativeType & derivativeM,
  DerivativeType & differential ) const
{
  /** Calculate the contributions to the derivatives with respect to each parameter. */
  if( nzji.size() == this->GetNumberOfParameters() )
  {
    /** Loop over all Jacobians. */
    typename DerivativeType::const_iterator imjacit  = imageJacobian.begin();
    typename DerivativeType::iterator derivativeFit  = derivativeF.begin();
    typename DerivativeType::iterator derivativeMit  = derivativeM.begin();
    typename DerivativeType::iterator differentialit = differential.begin();

    for( unsigned int mu = 0; mu < this->GetNumberOfParameters(); ++mu )
    {
      ( *derivativeFit )  += fixedImageValue * ( *imjacit );
      ( *derivativeMit )  += movingImageValue * ( *imjacit );
      ( *differentialit ) += ( *imjacit );
      ++imjacit;
      ++derivativeFit;
      ++derivativeMit;
      ++differentialit;
    }
  }
  else
  {
    /** Only pick the nonzero Jacobians. */
    for( unsigned int i = 0; i < imageJacobian.GetSize(); ++i )
    {
      const unsigned int index           = nzji[ i ];
      const RealType     differentialtmp = imageJacobian[ i ];
      derivativeF[ index ]  += fixedImageValue  * differentialtmp;
      derivativeM[ index ]  += movingImageValue * differentialtmp;
      differential[ index ] += differentialtmp;
    }
  }

} // end UpdateValueAndDerivativeTerms()


/**
 * ******************* GetValue *******************
 */

template< class TFixedImage, class TMovingImage >
typename MultiNormalizedCorrelationImageToImageMetric< TFixedImage, TMovingImage >::MeasureType
MultiNormalizedCorrelationImageToImageMetric< TFixedImage, TMovingImage >
::GetValue( const TransformParametersType & parameters ) const
{
  itkDebugMacro(<< "GetValue( " << parameters << " ) ");

  if (this->GetNumberOfFixedImages() != this->GetNumberOfMovingImages())
    itkExceptionMacro(<< "MultiNormalizedCorrelationImageToImageMetric requires the same number of fixed and moving images");

  /** Initialize some variables. */
  MeasureType measure = NumericTraits< MeasureType >::Zero;

  /** Call non-thread-safe stuff, such as:
   *   this->SetTransformParameters( parameters );
   *   this->GetImageSampler()->Update();
   * Because of these calls GetValueAndDerivative itself is not thread-safe,
   * so cannot be called multiple times simultaneously.
   * This is however needed in the CombinationImageToImageMetric.
   * In that case, you need to:
   * - switch the use of this function to on, using m_UseMetricSingleThreaded = true
   * - call BeforeThreadedGetValueAndDerivative once (single-threaded) before
   *   calling GetValueAndDerivative
   * - switch the use of this function to off, using m_UseMetricSingleThreaded = false
   * - Now you can call GetValueAndDerivative multi-threaded.
   */
  this->BeforeThreadedGetValueAndDerivative(parameters);

  /** Get the number of multiple images in the cluster */
  const unsigned int clusterSize = this->GetNumberOfFixedImages();

  /** Loop over all the multiple images in the cluster */
  for (unsigned int pos = 0; pos < clusterSize; ++pos)
  {
  this->m_NumberOfPixelsCounted = 0;

  /** Get a handle to the sample container. */
  ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();

  /** Create iterator over the sample container. */
  typename ImageSampleContainerType::ConstIterator fiter;
  typename ImageSampleContainerType::ConstIterator fbegin = sampleContainer->Begin();
  typename ImageSampleContainerType::ConstIterator fend = sampleContainer->End();

    /** Create variables to store intermediate results. */
    AccumulateType sff = NumericTraits< AccumulateType >::Zero;
    AccumulateType smm = NumericTraits< AccumulateType >::Zero;
    AccumulateType sfm = NumericTraits< AccumulateType >::Zero;
    AccumulateType sf = NumericTraits< AccumulateType >::Zero;
    AccumulateType sm = NumericTraits< AccumulateType >::Zero;

    /** Loop over the fixed image samples to calculate the mean squares. */
    for (fiter = fbegin; fiter != fend; ++fiter)
    {
      /** Read fixed coordinates and initialize some variables. */
      const FixedImagePointType& fixedPoint = (*fiter).Value().m_ImageCoordinates;
      RealType                    movingImageValue;
      MovingImagePointType        mappedPoint;

      /** Transform point and check if it is inside the B-spline support region. */
      bool sampleOk = this->TransformPoint(fixedPoint, mappedPoint);

      /** Check if point is inside mask. */
      if (sampleOk)
      {
        sampleOk = this->IsInsideMovingMask(mappedPoint);
      }

      /** Compute the moving image value and check if the point is
      * inside the moving image buffer. */
      if (sampleOk)
      {
        sampleOk = this->EvaluateMovingImageValueAndDerivative(
          mappedPoint, movingImageValue, 0, pos);
      }

      if (sampleOk)
      {
        this->m_NumberOfPixelsCounted++;

        /** Get the fixed image value. */
        FixedImageContinuousIndexType cindex;
        this->m_FixedImageInterpolatorVector[pos]->ConvertPointToContinuousIndex(fixedPoint, cindex);
        const RealType& fixedImageValue = this->GetFixedImageInterpolator(pos)->EvaluateAtContinuousIndex(cindex);

        /** Update some sums needed to calculate NC. */
        sff += fixedImageValue * fixedImageValue;
        smm += movingImageValue * movingImageValue;
        sfm += fixedImageValue * movingImageValue;
        if (this->m_SubtractMean)
        {
          sf += fixedImageValue;
          sm += movingImageValue;
        }

      } // end if sampleOk

    } // end for loop over the image sample container

    /** Check if enough samples were valid. */
    this->CheckNumberOfSamples(
      sampleContainer->Size(), this->m_NumberOfPixelsCounted);

    /** If SubtractMean, then subtract things from sff, smm and sfm. */
    const RealType N = static_cast<RealType>(this->m_NumberOfPixelsCounted);
    if (this->m_SubtractMean && this->m_NumberOfPixelsCounted > 0)
    {
      sff -= (sf * sf / N);
      smm -= (sm * sm / N);
      sfm -= (sf * sm / N);
    }

    /** The denominator of the NC. */
    const RealType denom = -1.0 * std::sqrt(sff * smm);

    /** Calculate the measure value. */
    if (this->m_NumberOfPixelsCounted > 0 && denom < -1e-14)
    {
      measure += sfm / denom;
    }
    else
    {
      measure += NumericTraits< MeasureType >::Zero;
    }
  } // end for loop over multiple images


   /** Check if enough samples were valid. */
   //this->CheckNumberOfSamples(
   //  sampleContainer->Size(), this->m_NumberOfPixelsCounted);

   // TODO not sure we need this
   ///** Compute average over variances. */
   //measure /= static_cast<float>(this->m_NumberOfPixelsCounted);
   ///** Normalize with initial variance. */
   //measure /= this->m_InitialVariance;


   /** Return the NC measure value. */
  return measure;

} // end GetValue()


/**
 * ******************* GetDerivative *******************
 */

template< class TFixedImage, class TMovingImage >
void
MultiNormalizedCorrelationImageToImageMetric< TFixedImage, TMovingImage >
::GetDerivative( const TransformParametersType & parameters,
  DerivativeType & derivative ) const
{
  /** When the derivative is calculated, all information for calculating
   * the metric value is available. It does not cost anything to calculate
   * the metric value now. Therefore, we have chosen to only implement the
   * GetValueAndDerivative(), supplying it with a dummy value variable.
   */
  MeasureType dummyvalue = NumericTraits< MeasureType >::Zero;
  this->GetValueAndDerivative( parameters, dummyvalue, derivative );

} // end GetDerivative()


/**
 * ******************* GetValueAndDerivativeSingleThreaded *******************
 */

template< class TFixedImage, class TMovingImage >
void
MultiNormalizedCorrelationImageToImageMetric< TFixedImage, TMovingImage >
::GetValueAndDerivative( const TransformParametersType & parameters,
  MeasureType & value, DerivativeType & derivative ) const
{
  itkDebugMacro( << "GetValueAndDerivative( " << parameters << " ) " );

  if (this->GetNumberOfFixedImages() != this->GetNumberOfMovingImages())
    itkExceptionMacro(<< "MultiNormalizedCorrelationImageToImageMetric requires the same number of fixed and moving images");

  typedef typename DerivativeType::ValueType DerivativeValueType;

  /** Get the number of multiple images in the cluster */
  const unsigned int clusterSize = this->GetNumberOfFixedImages();
  /** Loop over all the multiple images in the cluster */
  for (unsigned int pos = 0; pos < clusterSize; ++pos)
  {

  /** Initialize some variables. */
  this->m_NumberOfPixelsCounted = 0;
  derivative = DerivativeType(this->GetNumberOfParameters());
  derivative.Fill(NumericTraits< DerivativeValueType >::ZeroValue());
  DerivativeType derivativeF = DerivativeType(this->GetNumberOfParameters());
  derivativeF.Fill(NumericTraits< DerivativeValueType >::ZeroValue());
  DerivativeType derivativeM = DerivativeType(this->GetNumberOfParameters());
  derivativeM.Fill(NumericTraits< DerivativeValueType >::ZeroValue());
  DerivativeType differential = DerivativeType(this->GetNumberOfParameters());
  differential.Fill(NumericTraits< DerivativeValueType >::ZeroValue());

  /** Array that stores dM(x)/dmu, and the sparse Jacobian + indices. */
  NonZeroJacobianIndicesType nzji(this->m_AdvancedTransform->GetNumberOfNonZeroJacobianIndices());
  DerivativeType             imageJacobian(nzji.size());
  TransformJacobianType      jacobian;

  /** Initialize some variables for intermediate results. */
  AccumulateType sff = NumericTraits< AccumulateType >::Zero;
  AccumulateType smm = NumericTraits< AccumulateType >::Zero;
  AccumulateType sfm = NumericTraits< AccumulateType >::Zero;
  AccumulateType sf = NumericTraits< AccumulateType >::Zero;
  AccumulateType sm = NumericTraits< AccumulateType >::Zero;

  /** Call non-thread-safe stuff, such as:
   *   this->SetTransformParameters( parameters );
   *   this->GetImageSampler()->Update();
   * Because of these calls GetValueAndDerivative itself is not thread-safe,
   * so cannot be called multiple times simultaneously.
   * This is however needed in the CombinationImageToImageMetric.
   * In that case, you need to:
   * - switch the use of this function to on, using m_UseMetricSingleThreaded = true
   * - call BeforeThreadedGetValueAndDerivative once (single-threaded) before
   *   calling GetValueAndDerivative
   * - switch the use of this function to off, using m_UseMetricSingleThreaded = false
   * - Now you can call GetValueAndDerivative multi-threaded.
   */
  this->BeforeThreadedGetValueAndDerivative(parameters);

  /** Get a handle to the sample container. */
  ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();

  /** Create iterator over the sample container. */
  typename ImageSampleContainerType::ConstIterator fiter;
  typename ImageSampleContainerType::ConstIterator fbegin = sampleContainer->Begin();
  typename ImageSampleContainerType::ConstIterator fend = sampleContainer->End();

  /** Loop over the fixed image to calculate the correlation. */
  for (fiter = fbegin; fiter != fend; ++fiter)
  {
    /** Read fixed coordinates and initialize some variables. */
    const FixedImagePointType& fixedPoint = (*fiter).Value().m_ImageCoordinates;
    RealType                    movingImageValue;
    MovingImagePointType        mappedPoint;
    MovingImageDerivativeType   movingImageDerivative;

    /** Transform point and check if it is inside the B-spline support region. */
    bool sampleOk = this->TransformPoint(fixedPoint, mappedPoint);

    /** Check if point is inside mask. */
    if (sampleOk)
    {
      sampleOk = this->IsInsideMovingMask(mappedPoint);
    }

    /** Compute the moving image value M(T(x)) and derivative dM/dx and check if
     * the point is inside the moving image buffer.
     */
    if (sampleOk)
    {
      sampleOk = this->EvaluateMovingImageValueAndDerivative(
        mappedPoint, movingImageValue, &movingImageDerivative, pos);
    }

    if (sampleOk)
    {
      this->m_NumberOfPixelsCounted++;

      /** Get the fixed image value. */
      FixedImageContinuousIndexType cindex;
      this->m_FixedImageInterpolatorVector[pos]->ConvertPointToContinuousIndex(fixedPoint, cindex);
      const RealType& fixedImageValue = this->GetFixedImageInterpolator(pos)->EvaluateAtContinuousIndex(cindex);

      /** Get the TransformJacobian dT/dmu. */
      this->EvaluateTransformJacobian(fixedPoint, jacobian, nzji);

      /** Compute the innerproducts (dM/dx)^T (dT/dmu) and (dMask/dx)^T (dT/dmu). */
      this->EvaluateTransformJacobianInnerProduct(
        jacobian, movingImageDerivative, imageJacobian);

      /** Update some sums needed to calculate the value of NC. */
      sff += fixedImageValue * fixedImageValue;
      smm += movingImageValue * movingImageValue;
      sfm += fixedImageValue * movingImageValue;
      sf += fixedImageValue;  // Only needed when m_SubtractMean == true
      sm += movingImageValue; // Only needed when m_SubtractMean == true

      /** Compute this pixel's contribution to the derivative terms. */
      this->UpdateDerivativeTerms(
        fixedImageValue, movingImageValue, imageJacobian, nzji,
        derivativeF, derivativeM, differential);

    } // end if sampleOk

  } // end for loop over the image sample container

  /** Check if enough samples were valid. */
  this->CheckNumberOfSamples(
    sampleContainer->Size(), this->m_NumberOfPixelsCounted);

  /** If SubtractMean, then subtract things from sff, smm, sfm,
   * derivativeF and derivativeM.
   */
  const RealType N = static_cast<RealType>(this->m_NumberOfPixelsCounted);
  if (this->m_SubtractMean && this->m_NumberOfPixelsCounted > 0)
  {
    sff -= (sf * sf / N);
    smm -= (sm * sm / N);
    sfm -= (sf * sm / N);

    for (unsigned int i = 0; i < this->GetNumberOfParameters(); i++)
    {
      derivativeF[i] -= sf * differential[i] / N;
      derivativeM[i] -= sm * differential[i] / N;
    }
  }

  /** The denominator of the value and the derivative. */
  const RealType denom = -1.0 * std::sqrt(sff * smm);

  /** Calculate the value and the derivative. */
  if (this->m_NumberOfPixelsCounted > 0 && denom < -1e-14)
  {
    value = sfm / denom;
    for (unsigned int i = 0; i < this->GetNumberOfParameters(); i++)
    {
      derivative[i] += (derivativeF[i] - (sfm / smm) * derivativeM[i])
        / denom;
    }
  }
  else
  {
    value = NumericTraits< MeasureType >::Zero;
    derivative.Fill(NumericTraits< DerivativeValueType >::ZeroValue());
  }
  } // end for loop over multiple images
} // end GetValueAndDerivative()


} // end namespace itk

#endif // end #ifndef _itkMultiNormalizedCorrelationImageToImageMetric_hxx
