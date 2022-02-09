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
  itkDebugMacro(<< "GetValue( " << parameters << " ) ");

  if (this->GetNumberOfFixedImages() != this->GetNumberOfMovingImages())
    itkExceptionMacro(<< "MultiNormalizedCorrelationImageToImageMetric requires the same number of fixed and moving images");

  /** Initialize some variables. */
  this->m_NumberOfPixelsCounted = 0;
  MeasureType measure = NumericTraits< MeasureType >::Zero;

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

  /** Create list samples. */
  ListSamplePointer listSampleFixed = ListSampleType::New();
  ListSamplePointer listSampleMoving = ListSampleType::New();

  /** Compute the three list samples and the derivatives. */
  TransformJacobianContainerType        jacobianContainer;
  TransformJacobianIndicesContainerType jacobianIndicesContainer;
  SpatialDerivativeContainerType        spatialDerivativesContainer;
  this->ComputeListSampleValuesAndDerivativePlusJacobian(
    listSampleFixed, listSampleMoving,
    false, jacobianContainer, jacobianIndicesContainer, spatialDerivativesContainer);

  /** Temporary variables. */
  typedef typename NumericTraits< MeasureType >::AccumulateType AccumulateType;        // TODO probably do not need
  MeasurementVectorType z_F, z_M;

  /** Check if enough samples were valid. */
  unsigned long size = this->GetImageSampler()->GetOutput()->Size();
  this->CheckNumberOfSamples(size, this->m_NumberOfPixelsCounted);

  /** Loop over all query points, i.e. all samples. */
  for (unsigned long i = 0; i < this->m_NumberOfPixelsCounted; i++)
  {
    /** Get the fixed image value. */
    listSampleFixed->GetMeasurementVector(i, z_F);
    listSampleMoving->GetMeasurementVector(i, z_M);

    /** Update some sums needed to calculate the value of NC. */
    for (unsigned int j = 1; j < this->GetNumberOfFixedImages(); j++)
    {
      sff += z_F[j] * z_F[j];
      smm += z_M[j] * z_M[j];
      sfm += z_F[j] * z_M[j];
      sf += z_F[j];  // Only needed when m_SubtractMean == true
      sm += z_M[j]; // Only needed when m_SubtractMean == true
    }
  } // end for loop over the image sample container

  /** If SubtractMean, then subtract things from sff, smm, sfm,
   * derivativeF and derivativeM.
   */
  const RealType N = static_cast<RealType>(this->m_NumberOfPixelsCounted);
  if (this->m_SubtractMean && this->m_NumberOfPixelsCounted > 0)
  {
    sff -= (sf * sf / N);
    smm -= (sm * sm / N);
    sfm -= (sf * sm / N);
  }

  /** The denominator of the value and the derivative. */
  const RealType denom = -1.0 * std::sqrt(sff * smm);

  /** Calculate the value and the derivative. */
  if (this->m_NumberOfPixelsCounted > 0 && denom < -1e-14)
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

  if (this->GetNumberOfFixedImages() != this->GetNumberOfMovingImages())
    itkExceptionMacro(<< "MultiNormalizedCorrelationImageToImageMetric requires the same number of fixed and moving images");

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

  /** Create list samples. */
  ListSamplePointer listSampleFixed = ListSampleType::New();
  ListSamplePointer listSampleMoving = ListSampleType::New();

  /** Compute the three list samples and the derivatives. */
  TransformJacobianContainerType        jacobianContainer;
  TransformJacobianIndicesContainerType jacobianIndicesContainer;
  SpatialDerivativeContainerType        spatialDerivativesContainer;
  this->ComputeListSampleValuesAndDerivativePlusJacobian(
    listSampleFixed, listSampleMoving,
    true, jacobianContainer, jacobianIndicesContainer, spatialDerivativesContainer);

  /** Temporary variables. */
  typedef typename NumericTraits< MeasureType >::AccumulateType AccumulateType;
  MeasurementVectorType z_F, z_M;

  /** Check if enough samples were valid. */
  unsigned long size = this->GetImageSampler()->GetOutput()->Size();
  this->CheckNumberOfSamples(size, this->m_NumberOfPixelsCounted);

  /** Loop over all query points, i.e. all samples. */
  for (unsigned long i = 0; i < this->m_NumberOfPixelsCounted; i++)
  {
    /** Get the fixed image value. */
    listSampleFixed->GetMeasurementVector(i, z_F);
    listSampleMoving->GetMeasurementVector(i, z_M);

    /** Update some sums needed to calculate the value of NC. */
    sff += z_F[0] * z_F[0];
    smm += z_M[0] * z_M[0];
    sfm += z_F[0] * z_M[0];
    sf += z_F[0];  // Only needed when m_SubtractMean == true
    sm += z_M[0]; // Only needed when m_SubtractMean == true

    /** Compute this pixel's contribution to the derivative terms. */
    this->UpdateDerivativeTerms(
      z_F[0], z_M[0], imageJacobian, nzji, derivativeF, derivativeM, differential );

  } // end for loop over the image sample container

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

      /** Get and set the values of the fixed feature images. */
      for (unsigned int j = 1; j < this->GetNumberOfFixedImages(); j++)
      {
        fixedFeatureValue = this->m_FixedImageInterpolatorVector[j]
          ->Evaluate(fixedPoint);
        listSampleFixed->SetMeasurement(
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

} // end ComputeListSampleValuesAndDerivativePlusJacobian()

/**
 * ************************ EvaluateMovingFeatureImageDerivatives *************************
 */

template< class TFixedImage, class TMovingImage >
void
MultiNormalizedCorrelationImageToImageMetric< TFixedImage, TMovingImage >
::EvaluateMovingFeatureImageDerivatives(
  const MovingImagePointType& mappedPoint,
  SpatialDerivativeType& featureGradients) const
{
  /** Convert point to a continous index. */
  MovingImageContinuousIndexType cindex;
  this->m_Interpolator->ConvertPointToContinuousIndex(mappedPoint, cindex);

  /** Compute the spatial derivative for all feature images:
   * - either by calling a special function that only B-spline
   *   interpolators have,
   * - or by using a finite difference approximation of the
   *   pre-computed gradient images.
   * \todo: for now we only implement the first option.
   */
  if (this->m_InterpolatorsAreBSpline && !this->GetComputeGradient())
  {
    /** Computed moving image gradient using derivative B-spline kernel. */
    MovingImageDerivativeType gradient;
    for (unsigned int i = 1; i < this->GetNumberOfMovingImages(); ++i)
    {
      /** Compute the gradient at feature image i. */
      gradient = this->m_BSplineInterpolatorVector[i]
        ->EvaluateDerivativeAtContinuousIndex(cindex);

      /** Set the gradient into the Array2D. */
      featureGradients.set_row(i - 1, gradient.GetDataPointer());
    } // end for-loop
  } // end if
//  else
//  {
//  /** Get the gradient by NearestNeighboorInterpolation of the gradient image.
//  * It is assumed that the gradient image is computed beforehand.
//  */
//
//  /** Round the continuous index to the nearest neighbour. */
//  MovingImageIndexType index;
//  for ( unsigned int j = 0; j < MovingImageDimension; j++ )
//  {
//  index[ j ] = static_cast<long>( vnl_math::rnd( cindex[ j ] ) );
//  }
//
//  MovingImageDerivativeType gradient;
//  for ( unsigned int i = 0; i < this->m_NumberOfMovingFeatureImages; ++i )
//  {
//  /** Compute the gradient at feature image i. */
//  gradient = this->m_GradientFeatureImage[ i ]->GetPixel( index );
//
//  /** Set the gradient into the Array2D. */
//  featureGradients.set_column( i, gradient.GetDataPointer() );
//  } // end for-loop
//  } // end if

} // end EvaluateMovingFeatureImageDerivatives()

} // end namespace itk

#endif // end #ifndef _itkMultiNormalizedCorrelationImageToImageMetric_hxx
