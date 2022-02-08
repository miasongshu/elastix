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
  this->m_SubtractMean = false;

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
  itkDebugMacro( "GetValue( " << parameters << " ) " );

  /** Initialize some variables */
  this->m_NumberOfPixelsCounted = 0;
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
  this->BeforeThreadedGetValueAndDerivative( parameters );

  /** Get a handle to the sample container. */
  ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();

  /** Create iterator over the sample container. */
  typename ImageSampleContainerType::ConstIterator fiter;
  typename ImageSampleContainerType::ConstIterator fbegin = sampleContainer->Begin();
  typename ImageSampleContainerType::ConstIterator fend   = sampleContainer->End();

  /** Create variables to store intermediate results. */
  AccumulateType sff = NumericTraits< AccumulateType >::Zero;
  AccumulateType smm = NumericTraits< AccumulateType >::Zero;
  AccumulateType sfm = NumericTraits< AccumulateType >::Zero;
  AccumulateType sf  = NumericTraits< AccumulateType >::Zero;
  AccumulateType sm  = NumericTraits< AccumulateType >::Zero;

  /** Loop over the fixed image samples to calculate the mean squares. */
  for( fiter = fbegin; fiter != fend; ++fiter )
  {
    /** Read fixed coordinates and initialize some variables. */
    const FixedImagePointType & fixedPoint = ( *fiter ).Value().m_ImageCoordinates;
    RealType                    movingImageValue;
    MovingImagePointType        mappedPoint;

    /** Transform point and check if it is inside the B-spline support region. */
    bool sampleOk = this->TransformPoint( fixedPoint, mappedPoint );

    /** Check if point is inside mask. */
    if( sampleOk )
    {
      sampleOk = this->IsInsideMovingMask( mappedPoint );
    }

    /** Compute the moving image value and check if the point is
    * inside the moving image buffer. */
    if( sampleOk )
    {
      sampleOk = this->EvaluateMovingImageValueAndDerivative(
        mappedPoint, movingImageValue, 0 );
    }

    if( sampleOk )
    {
      this->m_NumberOfPixelsCounted++;

      /** Get the fixed image value. */
      const RealType & fixedImageValue = static_cast< double >( ( *fiter ).Value().m_ImageValue );

      /** Update some sums needed to calculate NC. */
      sff += fixedImageValue  * fixedImageValue;
      smm += movingImageValue * movingImageValue;
      sfm += fixedImageValue  * movingImageValue;
      if( this->m_SubtractMean )
      {
        sf += fixedImageValue;
        sm += movingImageValue;
      }

    } // end if sampleOk

  } // end for loop over the image sample container

  /** Check if enough samples were valid. */
  this->CheckNumberOfSamples(
    sampleContainer->Size(), this->m_NumberOfPixelsCounted );

  /** If SubtractMean, then subtract things from sff, smm and sfm. */
  const RealType N = static_cast< RealType >( this->m_NumberOfPixelsCounted );
  if( this->m_SubtractMean && this->m_NumberOfPixelsCounted > 0 )
  {
    sff -= ( sf * sf / N );
    smm -= ( sm * sm / N );
    sfm -= ( sf * sm / N );
  }

  /** The denominator of the NC. */
  const RealType denom = -1.0 * std::sqrt( sff * smm );

  /** Calculate the measure value. */
  if( this->m_NumberOfPixelsCounted > 0 && denom < -1e-14 )
  {
    measure = sfm / denom;
  }
  else
  {
    measure = NumericTraits< MeasureType >::Zero;
  }

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

  typedef typename DerivativeType::ValueType DerivativeValueType;

  /** Initialize some variables. */
  this->m_NumberOfPixelsCounted = 0;
  derivative                    = DerivativeType( this->GetNumberOfParameters() );
  derivative.Fill( NumericTraits< DerivativeValueType >::ZeroValue() );
  DerivativeType derivativeF = DerivativeType( this->GetNumberOfParameters() );
  derivativeF.Fill( NumericTraits< DerivativeValueType >::ZeroValue() );
  DerivativeType derivativeM = DerivativeType( this->GetNumberOfParameters() );
  derivativeM.Fill( NumericTraits< DerivativeValueType >::ZeroValue() );
  DerivativeType differential = DerivativeType( this->GetNumberOfParameters() );
  differential.Fill( NumericTraits< DerivativeValueType >::ZeroValue() );

  /** Array that stores dM(x)/dmu, and the sparse Jacobian + indices. */
  NonZeroJacobianIndicesType nzji( this->m_AdvancedTransform->GetNumberOfNonZeroJacobianIndices() );
  DerivativeType             imageJacobian( nzji.size() );
  TransformJacobianType      jacobian;

  /** Initialize some variables for intermediate results. */
  AccumulateType sff = NumericTraits< AccumulateType >::Zero;
  AccumulateType smm = NumericTraits< AccumulateType >::Zero;
  AccumulateType sfm = NumericTraits< AccumulateType >::Zero;
  AccumulateType sf  = NumericTraits< AccumulateType >::Zero;
  AccumulateType sm  = NumericTraits< AccumulateType >::Zero;

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
  this->BeforeThreadedGetValueAndDerivative( parameters );

  /** Get a handle to the sample container. */
  ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();

  /** Create iterator over the sample container. */
  typename ImageSampleContainerType::ConstIterator fiter;
  typename ImageSampleContainerType::ConstIterator fbegin = sampleContainer->Begin();
  typename ImageSampleContainerType::ConstIterator fend   = sampleContainer->End();

  /** Loop over the fixed image to calculate the correlation. */
  for( fiter = fbegin; fiter != fend; ++fiter )
  {
    /** Read fixed coordinates and initialize some variables. */
    const FixedImagePointType & fixedPoint = ( *fiter ).Value().m_ImageCoordinates;
    RealType                    movingImageValue;
    MovingImagePointType        mappedPoint;
    MovingImageDerivativeType   movingImageDerivative;

    /** Transform point and check if it is inside the B-spline support region. */
    bool sampleOk = this->TransformPoint( fixedPoint, mappedPoint );

    /** Check if point is inside mask. */
    if( sampleOk )
    {
      sampleOk = this->IsInsideMovingMask( mappedPoint );
    }

    /** Compute the moving image value M(T(x)) and derivative dM/dx and check if
     * the point is inside the moving image buffer.
     */
    if( sampleOk )
    {
      sampleOk = this->EvaluateMovingImageValueAndDerivative(
        mappedPoint, movingImageValue, &movingImageDerivative );
    }

    if( sampleOk )
    {
      this->m_NumberOfPixelsCounted++;

      /** Get the fixed image value. */
      const RealType & fixedImageValue = static_cast< RealType >( ( *fiter ).Value().m_ImageValue );

      /** Get the TransformJacobian dT/dmu. */
      this->EvaluateTransformJacobian( fixedPoint, jacobian, nzji );

      /** Compute the innerproducts (dM/dx)^T (dT/dmu) and (dMask/dx)^T (dT/dmu). */
      this->EvaluateTransformJacobianInnerProduct(
        jacobian, movingImageDerivative, imageJacobian );

      /** Update some sums needed to calculate the value of NC. */
      sff += fixedImageValue  * fixedImageValue;
      smm += movingImageValue * movingImageValue;
      sfm += fixedImageValue  * movingImageValue;
      sf  += fixedImageValue;  // Only needed when m_SubtractMean == true
      sm  += movingImageValue; // Only needed when m_SubtractMean == true

      /** Compute this pixel's contribution to the derivative terms. */
      this->UpdateDerivativeTerms(
        fixedImageValue, movingImageValue, imageJacobian, nzji,
        derivativeF, derivativeM, differential );

    } // end if sampleOk

  } // end for loop over the image sample container

  /** Check if enough samples were valid. */
  this->CheckNumberOfSamples(
    sampleContainer->Size(), this->m_NumberOfPixelsCounted );

  /** If SubtractMean, then subtract things from sff, smm, sfm,
   * derivativeF and derivativeM.
   */
  const RealType N = static_cast< RealType >( this->m_NumberOfPixelsCounted );
  if( this->m_SubtractMean && this->m_NumberOfPixelsCounted > 0 )
  {
    sff -= ( sf * sf / N );
    smm -= ( sm * sm / N );
    sfm -= ( sf * sm / N );

    for( unsigned int i = 0; i < this->GetNumberOfParameters(); i++ )
    {
      derivativeF[ i ] -= sf * differential[ i ] / N;
      derivativeM[ i ] -= sm * differential[ i ] / N;
    }
  }

  /** The denominator of the value and the derivative. */
  const RealType denom = -1.0 * std::sqrt( sff * smm );

  /** Calculate the value and the derivative. */
  if( this->m_NumberOfPixelsCounted > 0 && denom < -1e-14 )
  {
    value = sfm / denom;
    for( unsigned int i = 0; i < this->GetNumberOfParameters(); i++ )
    {
      derivative[ i ] = ( derivativeF[ i ] - ( sfm / smm ) * derivativeM[ i ] )
        / denom;
    }
  }
  else
  {
    value = NumericTraits< MeasureType >::Zero;
    derivative.Fill( NumericTraits< DerivativeValueType >::ZeroValue() );
  }

} // end GetValueAndDerivative()

/**
 * ************************ ComputeListSampleValuesAndDerivativePlusJacobian *************************
 */

template< class TFixedImage, class TMovingImage >
void
MultiNormalizedCorrelationImageToImageMetric< TFixedImage, TMovingImage >
::ComputeListSampleValuesAndDerivativePlusJacobian(
  const ListSamplePointer& listSampleFixed,
  const ListSamplePointer& listSampleMoving,
  const ListSamplePointer& listSampleJoint,
  const bool& doDerivative,
  TransformJacobianContainerType& jacobianContainer,
  TransformJacobianIndicesContainerType& jacobianIndicesContainer,
  SpatialDerivativeContainerType& spatialDerivativesContainer) const
{
  /** Initialize. */
  this->m_NumberOfPixelsCounted = 0;
  jacobianContainer.resize(0);
  jacobianIndicesContainer.resize(0);
  spatialDerivativesContainer.resize(0);

  /** Get a handle to the sample container. */
  ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();
  const unsigned long         nrOfRequestedSamples = sampleContainer->Size();

  /** Create an iterator over the sample container. */
  typename ImageSampleContainerType::ConstIterator fiter;
  typename ImageSampleContainerType::ConstIterator fbegin = sampleContainer->Begin();
  typename ImageSampleContainerType::ConstIterator fend = sampleContainer->End();

  /** Get the size of the feature vectors. */
  const unsigned int fixedSize = this->GetNumberOfFixedImages();
  const unsigned int movingSize = this->GetNumberOfMovingImages();
  const unsigned int jointSize = fixedSize + movingSize;

  /** Resize the list samples so that enough memory is allocated. */
  listSampleFixed->SetMeasurementVectorSize(fixedSize);
  listSampleFixed->Resize(nrOfRequestedSamples);
  listSampleMoving->SetMeasurementVectorSize(movingSize);
  listSampleMoving->Resize(nrOfRequestedSamples);
  listSampleJoint->SetMeasurementVectorSize(jointSize);
  listSampleJoint->Resize(nrOfRequestedSamples);

  /** Potential speedup: it avoids re-allocations. I noticed performance
   * gains when nrOfRequestedSamples is about 10000 or higher.
   */
  jacobianContainer.reserve(nrOfRequestedSamples);
  jacobianIndicesContainer.reserve(nrOfRequestedSamples);
  spatialDerivativesContainer.reserve(nrOfRequestedSamples);

  /** Create variables to store intermediate results. */
  RealType                   movingImageValue;
  MovingImagePointType       mappedPoint;
  double                     fixedFeatureValue = 0.0;
  double                     movingFeatureValue = 0.0;
  NonZeroJacobianIndicesType nzji(
    this->m_AdvancedTransform->GetNumberOfNonZeroJacobianIndices());
  TransformJacobianType jacobian;

  /** Loop over the fixed image samples to calculate the list samples. */
  unsigned int ii = 0;
  for (fiter = fbegin; fiter != fend; ++fiter)
  {
    /** Read fixed coordinates and initialize some variables. */
    const FixedImagePointType& fixedPoint = (*fiter).Value().m_ImageCoordinates;

    /** Transform point and check if it is inside the B-spline support region. */
    bool sampleOk = this->TransformPoint(fixedPoint, mappedPoint);

    /** Check if point is inside all moving masks. */
    if (sampleOk)
    {
      sampleOk = this->IsInsideMovingMask(mappedPoint);
    }

    /** Compute the moving image value M(T(x)) and possibly the
     * derivative dM/dx and check if the point is inside all
     * moving images buffers.
     */
    MovingImageDerivativeType movingImageDerivative;
    if (sampleOk)
    {
      if (doDerivative)
      {
        sampleOk = this->EvaluateMovingImageValueAndDerivative(
          mappedPoint, movingImageValue, &movingImageDerivative);
      }
      else
      {
        sampleOk = this->EvaluateMovingImageValueAndDerivative(
          mappedPoint, movingImageValue, 0);
      }
    }

    /** This is a valid sample: in this if-statement the actual
     * addition to the list samples is done.
     */
    if (sampleOk)
    {
      /** Get the fixed image value. */
      const RealType& fixedImageValue = static_cast<RealType>(
        (*fiter).Value().m_ImageValue);

      /** Add the samples to the ListSampleCarrays. */
      listSampleFixed->SetMeasurement(this->m_NumberOfPixelsCounted, 0,
        fixedImageValue);
      listSampleMoving->SetMeasurement(this->m_NumberOfPixelsCounted, 0,
        movingImageValue);
      listSampleJoint->SetMeasurement(this->m_NumberOfPixelsCounted, 0,
        fixedImageValue);
      listSampleJoint->SetMeasurement(this->m_NumberOfPixelsCounted,
        this->GetNumberOfFixedImages(), movingImageValue);

      /** Get and set the values of the fixed feature images. */
      for (unsigned int j = 1; j < this->GetNumberOfFixedImages(); j++)
      {
        fixedFeatureValue = this->m_FixedImageInterpolatorVector[j]
          ->Evaluate(fixedPoint);
        listSampleFixed->SetMeasurement(
          this->m_NumberOfPixelsCounted, j, fixedFeatureValue);
        listSampleJoint->SetMeasurement(
          this->m_NumberOfPixelsCounted, j, fixedFeatureValue);
      }

      /** Get and set the values of the moving feature images. */
      for (unsigned int j = 1; j < this->GetNumberOfMovingImages(); j++)
      {
        movingFeatureValue = this->m_InterpolatorVector[j]
          ->Evaluate(mappedPoint);
        listSampleMoving->SetMeasurement(
          this->m_NumberOfPixelsCounted,
          j,
          movingFeatureValue);
        listSampleJoint->SetMeasurement(
          this->m_NumberOfPixelsCounted,
          j + this->GetNumberOfFixedImages(),
          movingFeatureValue);
      }

      /** Compute additional stuff for the computation of the derivative, if necessary.
       * - the Jacobian of the transform: dT/dmu(x_i).
       * - the spatial derivative of all moving feature images: dz_q^m/dx(T(x_i)).
       */
      if (doDerivative)
      {
        /** Get the TransformJacobian dT/dmu. */
        this->EvaluateTransformJacobian(fixedPoint, jacobian, nzji);
        jacobianContainer.push_back(jacobian);
        jacobianIndicesContainer.push_back(nzji);

        /** Get the spatial derivative of the moving image. */
        SpatialDerivativeType spatialDerivatives(
          this->GetNumberOfMovingImages(),
          this->FixedImageDimension);
        spatialDerivatives.set_row(0, movingImageDerivative.GetDataPointer());

        /** Get the spatial derivatives of the moving feature images. */
        SpatialDerivativeType movingFeatureImageDerivatives(
          this->GetNumberOfMovingImages() - 1,
          this->FixedImageDimension);
        this->EvaluateMovingFeatureImageDerivatives(
          mappedPoint, movingFeatureImageDerivatives);
        spatialDerivatives.update(movingFeatureImageDerivatives, 1, 0);

        /** Put the spatial derivatives of this sample into the container. */
        spatialDerivativesContainer.push_back(spatialDerivatives);

      } // end if doDerivative

      /** Update the NumberOfPixelsCounted. */
      this->m_NumberOfPixelsCounted++;

      ii++;

    } // end if sampleOk

  } // end for loop over the image sample container

  /** The listSamples are of size sampleContainer->Size(). However, not all of
   * those points made it to the respective list samples. Therefore, we set
   * the actual number of pixels in the sample container, so that the binary
   * trees know where to loop over. This must not be forgotten!
   */
  listSampleFixed->SetActualSize(this->m_NumberOfPixelsCounted);
  listSampleMoving->SetActualSize(this->m_NumberOfPixelsCounted);
  listSampleJoint->SetActualSize(this->m_NumberOfPixelsCounted);

} // end ComputeListSampleValuesAndDerivativePlusJacobian()

} // end namespace itk

#endif // end #ifndef _itkMultiNormalizedCorrelationImageToImageMetric_hxx
