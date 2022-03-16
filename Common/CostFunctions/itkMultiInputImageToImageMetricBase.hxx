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
#ifndef _itkMultiInputImageToImageMetricBase_hxx
#define _itkMultiInputImageToImageMetricBase_hxx

#include "itkMultiInputImageToImageMetricBase.h"

#include "itkHardLimiterFunction.h"
#include "itkExponentialLimiterFunction.h"

/** Macros to reduce some copy-paste work.
 * These macros provide the implementation of
 * all Set/GetFixedImage, Set/GetInterpolator etc methods
 *
 * The macros are undef'ed at the end of this file
 */

/** Macro for setting objects. */
#define itkImplementationSetObjectMacro( _name, _type ) \
  template< class TFixedImage, class TMovingImage > \
  void \
  MultiInputImageToImageMetricBase< TFixedImage, TMovingImage > \
  ::Set##_name( _type * _arg, unsigned int pos ) \
  { \
    if( this->m_##_name##Vector.size() < pos + 1 ) \
    { \
      this->m_##_name##Vector.resize( pos + 1 ); \
      this->m_NumberOf##_name##s = pos + 1; \
    } \
    if( pos == 0 ) \
    { \
      this->Superclass::Set##_name( _arg ); \
    } \
    if( this->m_##_name##Vector[ pos ] != _arg ) \
    { \
      this->m_##_name##Vector[ pos ] = _arg; \
      this->Modified(); \
    } \
  } // comments for allowing ; after calling the macro

/** Macro for setting objects. */
#define itkImplementationSetObjectMacro2( _name, _type ) \
  template< class TFixedImage, class TMovingImage > \
  void \
  MultiInputImageToImageMetricBase< TFixedImage, TMovingImage > \
  ::Set##_name( _type * _arg, unsigned int pos ) \
  { \
    if( this->m_##_name##Vector.size() < pos + 1 ) \
    { \
      this->m_##_name##Vector.resize( pos + 1 ); \
      this->m_NumberOf##_name##s = pos + 1; \
    } \
    if( this->m_##_name##Vector[ pos ] != _arg ) \
    { \
      this->m_##_name##Vector[ pos ] = _arg; \
      this->Modified(); \
    } \
  } // comments for allowing ; after calling the macro

/** Macro for getting objects. */
#define itkImplementationGetObjectMacro( _name, _type ) \
  template< class TFixedImage, class TMovingImage > \
  typename MultiInputImageToImageMetricBase< TFixedImage, TMovingImage >::_type   \
  * MultiInputImageToImageMetricBase< TFixedImage, TMovingImage > \
  ::Get##_name( unsigned int pos ) const \
  { \
    if( this->m_##_name##Vector.size() < pos + 1 ) \
    { \
      return 0; \
    } \
    return this->m_##_name##Vector[ pos ]; \
  } // comments for allowing ; after calling the macro

/** Macro for getting const objects. */
#define itkImplementationGetConstObjectMacro( _name, _type ) \
  template< class TFixedImage, class TMovingImage > \
  const typename MultiInputImageToImageMetricBase< TFixedImage, TMovingImage >::_type   \
  * MultiInputImageToImageMetricBase< TFixedImage, TMovingImage > \
  ::Get##_name( unsigned int pos ) const \
  { \
    if( this->m_##_name##Vector.size() < pos + 1 ) \
    { \
      return 0; \
    } \
    return this->m_##_name##Vector[ pos ]; \
  } // comments for allowing ; after calling the macro

namespace itk
{

/**
 * ************************ Constructor *************************
 */

template< class TFixedImage, class TMovingImage >
MultiInputImageToImageMetricBase< TFixedImage, TMovingImage >
::MultiInputImageToImageMetricBase()
{
  this->m_NumberOfFixedImages             = 0;
  this->m_NumberOfFixedImageMasks         = 0;
  this->m_NumberOfFixedImageRegions       = 0;
  this->m_NumberOfMovingImages            = 0;
  this->m_NumberOfMovingImageMasks        = 0;
  this->m_NumberOfInterpolators           = 0;
  this->m_NumberOfFixedImageInterpolators = 0;
  this->m_NumberOfFixedImageLimiters      = 0;
  this->m_NumberOfMovingImageLimiters     = 0;

  this->m_InterpolatorsAreBSpline = false;

} // end Constructor()


/** Set components. */
itkImplementationSetObjectMacro( FixedImage, const FixedImageType );
itkImplementationSetObjectMacro( FixedImageMask, FixedImageMaskType );
itkImplementationSetObjectMacro( MovingImage, const MovingImageType );
itkImplementationSetObjectMacro( MovingImageMask, MovingImageMaskType );
itkImplementationSetObjectMacro( Interpolator, InterpolatorType );
itkImplementationSetObjectMacro2( FixedImageInterpolator, FixedImageInterpolatorType );
itkImplementationSetObjectMacro( FixedImageLimiter, FixedImageLimiterType );
itkImplementationSetObjectMacro( MovingImageLimiter, MovingImageLimiterType );


/** Get components. */
itkImplementationGetConstObjectMacro( FixedImage, FixedImageType );
itkImplementationGetObjectMacro( FixedImageMask, FixedImageMaskType );
itkImplementationGetConstObjectMacro( MovingImage, MovingImageType );
itkImplementationGetObjectMacro( MovingImageMask, MovingImageMaskType );
itkImplementationGetObjectMacro( Interpolator, InterpolatorType );
itkImplementationGetObjectMacro( FixedImageInterpolator, FixedImageInterpolatorType );
itkImplementationGetObjectMacro( BSplineInterpolator, BSplineInterpolatorType );
itkImplementationGetConstObjectMacro( FixedImageLimiter, FixedImageLimiterType );
itkImplementationGetConstObjectMacro( MovingImageLimiter, MovingImageLimiterType );

/**
 * ************************ SetFixedImageRegion *************************
 */

template< class TFixedImage, class TMovingImage >
void
MultiInputImageToImageMetricBase< TFixedImage, TMovingImage >
::SetFixedImageRegion( const FixedImageRegionType _arg, unsigned int pos )
{
  if( this->m_FixedImageRegionVector.size() < pos + 1 )
  {
    this->m_FixedImageRegionVector.resize( pos + 1 );
    this->m_NumberOfFixedImageRegions = pos + 1;
  }
  if( pos == 0 )
  {
    this->Superclass::SetFixedImageRegion( _arg );
  }
  if( this->m_FixedImageRegionVector[ pos ] != _arg )
  {
    this->m_FixedImageRegionVector[ pos ] = _arg;
    this->Modified();
  }

} // end SetFixedImageRegion()


/**
 * ************************ GetFixedImageRegion *************************
 */

template< class TFixedImage, class TMovingImage >
const typename MultiInputImageToImageMetricBase< TFixedImage, TMovingImage >
::FixedImageRegionType
& MultiInputImageToImageMetricBase< TFixedImage, TMovingImage >
::GetFixedImageRegion( unsigned int pos ) const
{
  if( this->m_FixedImageRegionVector.size() < pos )
  {
    return this->m_DummyFixedImageRegion;
  }

  return this->m_FixedImageRegionVector[ pos ];

} // end GetFixedImageRegion()


/**
 * ****************** CheckForBSplineInterpolators **********************
 */

template< class TFixedImage, class TMovingImage >
void
MultiInputImageToImageMetricBase< TFixedImage, TMovingImage >
::CheckForBSplineInterpolators( void )
{
  /** Check if the interpolators are of type BSplineInterpolateImageFunction.
   * If so, we can make use of its EvaluateDerivatives method.
   * Otherwise, an exception is thrown.
   */
  this->m_InterpolatorsAreBSpline = true;
  this->m_BSplineInterpolatorVector.resize( this->m_NumberOfMovingImages );

  for( unsigned int i = 0; i < this->m_NumberOfMovingImages; ++i )
  {
    BSplineInterpolatorType * testPtr
      = dynamic_cast< BSplineInterpolatorType * >(
      this->m_InterpolatorVector[ i ].GetPointer() );

    if( testPtr )
    {
      this->m_InterpolatorsAreBSpline       &= true;
      this->m_BSplineInterpolatorVector[ i ] = testPtr;
      itkDebugMacro( << "Interpolator " << i << " is B-spline." );
    }
    else
    {
      this->m_InterpolatorsAreBSpline &= false;
      itkDebugMacro( << "Interpolator " << i << " is NOT B-spline." );
      itkExceptionMacro( << "Interpolator " << i << " is NOT B-spline." );
    }
  } // end for-loop

} // end CheckForBSplineInterpolators()


/**
 * ****************** Initialize **********************
 */

template< class TFixedImage, class TMovingImage >
void
MultiInputImageToImageMetricBase< TFixedImage, TMovingImage >
::Initialize( void )
{
  /** Connect the interpolators. */
  for( unsigned int i = 0; i < this->GetNumberOfInterpolators(); ++i )
  {
    this->m_InterpolatorVector[ i ]->SetInputImage( this->m_MovingImageVector[ i ] );
  }

  /** Connect the fixed image interpolators. */
  for( unsigned int i = 0; i < this->GetNumberOfFixedImageInterpolators(); ++i )
  {
    this->m_FixedImageInterpolatorVector[ i ]->SetInputImage( this->m_FixedImageVector[ i ] );
  }

  /** Check for B-spline interpolators. */
  this->CheckForBSplineInterpolators();

  /** Call the superclass' implementation. */
  this->Superclass::Initialize();

} // end Initialize()

/**
 * ****************** InitializeLimiters *****************************
 */

template< class TFixedImage, class TMovingImage >
void
MultiInputImageToImageMetricBase< TFixedImage, TMovingImage >
::InitializeLimiters(void)
{
  /** Set up fixed limiter. */
  if (this->GetUseFixedImageLimiter())
  {
    if (this->GetFixedImageLimiter() == 0)
    {
      itkExceptionMacro(<< "No fixed image limiter has been set!");
    }

    for (unsigned int i = 0; i < this->GetNumberOfFixedImages(); ++i)
    {
      itk::TimeProbe timer;
      timer.Start();

      typedef typename itk::ComputeImageExtremaFilter<FixedImageType> ComputeFixedImageExtremaFilterType;
      typename ComputeFixedImageExtremaFilterType::Pointer computeFixedImageExtrema
        = ComputeFixedImageExtremaFilterType::New();
      computeFixedImageExtrema->SetInput(this->GetFixedImage(i));  
      computeFixedImageExtrema->SetImageRegion(this->GetFixedImageRegion(i));

      if (this->GetNumberOfFixedImageMasks() > i
        && this->m_FixedImageMaskVector[i].IsNotNull())
      {
        computeFixedImageExtrema->SetUseMask(true);

        const FixedImageMaskSpatialObject2Type* fMask
          = dynamic_cast<const FixedImageMaskSpatialObject2Type*>(this->m_FixedImageMaskVector[i].GetPointer());
        if (fMask)
        {
          computeFixedImageExtrema->SetImageSpatialMask(fMask);
        }
        else
        {
          computeFixedImageExtrema->SetImageMask(this->GetFixedImageMask(i));
        }
      }

      computeFixedImageExtrema->Update();
      timer.Stop();
      elxout << "  Computing the fixed image #" << i << "  extrema took "
        << static_cast<long>(timer.GetMean() * 1000) << " ms." << std::endl;

      this->m_FixedImageTrueMax = computeFixedImageExtrema->GetMaximum();
      this->m_FixedImageTrueMin = computeFixedImageExtrema->GetMinimum();

      this->m_FixedImageMinLimit = static_cast<FixedImageLimiterOutputType>(
        this->m_FixedImageTrueMin - this->m_FixedLimitRangeRatio * (this->m_FixedImageTrueMax - this->m_FixedImageTrueMin));
      this->m_FixedImageMaxLimit = static_cast<FixedImageLimiterOutputType>(
        this->m_FixedImageTrueMax + this->m_FixedLimitRangeRatio * (this->m_FixedImageTrueMax - this->m_FixedImageTrueMin));

      typedef itk::HardLimiterFunction< RealType, FixedImageDimension >         FixedLimiterType;
      auto fixedLimiter = FixedLimiterType::New();

      fixedLimiter->SetLowerThreshold(static_cast<RealType>(this->m_FixedImageTrueMin));
      fixedLimiter->SetUpperThreshold(static_cast<RealType>(this->m_FixedImageTrueMax));
      fixedLimiter->SetLowerBound(this->m_FixedImageMinLimit);
      fixedLimiter->SetUpperBound(this->m_FixedImageMaxLimit);

      fixedLimiter->Initialize();

      this->SetFixedImageLimiter(fixedLimiter, i);
    }
  }

  /** Set up moving limiter. */
  if (this->GetUseMovingImageLimiter())
  {
    if (this->GetMovingImageLimiter() == 0)
    {
      itkExceptionMacro(<< "No moving image limiter has been set!");
    }

    for (unsigned int i = 0; i < this->GetNumberOfMovingImages(); ++i)
    {
      itk::TimeProbe timer;
      timer.Start();

      typedef typename itk::ComputeImageExtremaFilter<MovingImageType> ComputeMovingImageExtremaFilterType;
      typename ComputeMovingImageExtremaFilterType::Pointer computeMovingImageExtrema
        = ComputeMovingImageExtremaFilterType::New();
      computeMovingImageExtrema->SetInput(this->GetMovingImage(i));
      computeMovingImageExtrema->SetImageRegion(this->GetMovingImage(i)->GetBufferedRegion());
      if (this->GetNumberOfMovingImageMasks() > i
        && this->m_MovingImageMaskVector[i].IsNotNull())
      {
        computeMovingImageExtrema->SetUseMask(true);
        const MovingImageMaskSpatialObject2Type* mMask
          = dynamic_cast<const MovingImageMaskSpatialObject2Type*>(this->m_MovingImageMaskVector[i].GetPointer());
        if (mMask)
        {
          computeMovingImageExtrema->SetImageSpatialMask(mMask);
        }
        else
        {
          computeMovingImageExtrema->SetImageMask(this->GetMovingImageMask(i));
        }
      }
      computeMovingImageExtrema->Update();

      timer.Stop();
      elxout << "  Computing the moving image #" << i << " extrema took "
        << static_cast<long>(timer.GetMean() * 1000) << " ms." << std::endl;

      // TODO see comment above
      this->m_MovingImageTrueMax = computeMovingImageExtrema->GetMaximum();
      this->m_MovingImageTrueMin = computeMovingImageExtrema->GetMinimum();

      this->m_MovingImageMinLimit = static_cast<MovingImageLimiterOutputType>(
        this->m_MovingImageTrueMin - this->m_MovingLimitRangeRatio * (this->m_MovingImageTrueMax - this->m_MovingImageTrueMin));
      this->m_MovingImageMaxLimit = static_cast<MovingImageLimiterOutputType>(
        this->m_MovingImageTrueMax + this->m_MovingLimitRangeRatio * (this->m_MovingImageTrueMax - this->m_MovingImageTrueMin));

      typedef itk::ExponentialLimiterFunction< RealType, MovingImageDimension > MovingLimiterType;
      auto movingLimiter = MovingLimiterType::New();

      movingLimiter->SetLowerThreshold(static_cast<RealType>(this->m_MovingImageTrueMin));
      movingLimiter->SetUpperThreshold(static_cast<RealType>(this->m_MovingImageTrueMax));
      movingLimiter->SetLowerBound(this->m_MovingImageMinLimit);
      movingLimiter->SetUpperBound(this->m_MovingImageMaxLimit);

      movingLimiter->Initialize();

      this->SetMovingImageLimiter(movingLimiter, i);
    }
  }

} // end InitializeLimiters()

/**
 * ********************* InitializeImageSampler ****************************
 */

template< class TFixedImage, class TMovingImage >
void
MultiInputImageToImageMetricBase< TFixedImage, TMovingImage >
::InitializeImageSampler( void )
{
  if( this->GetUseImageSampler() )
  {
    /** Check if the ImageSampler is set. */
    if( !this->m_ImageSampler )
    {
      itkExceptionMacro( << "ImageSampler is not present" );
    }

    /** Initialize the Image Sampler: set the fixed images. */
    for( unsigned int i = 0; i < this->GetNumberOfFixedImages(); ++i )
    {
      this->m_ImageSampler->SetInput( i, this->m_FixedImageVector[ i ] );
    }

    /** Initialize the Image Sampler: set the fixed image masks. */
    for( unsigned int i = 0; i < this->GetNumberOfFixedImageMasks(); ++i )
    {
      this->m_ImageSampler->SetMask( this->m_FixedImageMaskVector[ i ], i );
    }

    /** Initialize the Image Sampler: set the fixed image regions. */
    for( unsigned int i = 0; i < this->GetNumberOfFixedImages(); ++i )
    {
      this->m_ImageSampler->SetInputImageRegion( this->m_FixedImageRegionVector[ i ], i );
    }
  }

} // end InitializeImageSampler()


/**
 * ******************* EvaluateMovingImageValueAndDerivative ******************
 */

template< class TFixedImage, class TMovingImage >
bool
MultiInputImageToImageMetricBase< TFixedImage, TMovingImage >
::EvaluateMovingImageValueAndDerivative(
  const MovingImagePointType & mappedPoint,
  RealType & movingImageValue,
  MovingImageDerivativeType * gradient ) const
{
  /** Check if the mapped point is inside the moving image buffers of the feature images. */
  bool sampleOk = true;
  for( unsigned int i = 1; i < this->GetNumberOfInterpolators(); ++i )
  {
    sampleOk &= this->GetInterpolator( i )->IsInsideBuffer( mappedPoint );

    /** If not inside this buffer we can quit. */
    if( !sampleOk ) { return false; }
  }

  /** Compute value and possibly derivative of the moving image. */
  return this->Superclass::EvaluateMovingImageValueAndDerivative(
    mappedPoint, movingImageValue, gradient );

} // end EvaluateMovingImageValueAndDerivative()


/**
 * ******************* EvaluateMovingImageValueAndDerivative ******************
 */

template< class TFixedImage, class TMovingImage >
bool
MultiInputImageToImageMetricBase< TFixedImage, TMovingImage >
::EvaluateMovingImageValueAndDerivative(
  const MovingImagePointType& mappedPoint,
  RealType& movingImageValue,
  MovingImageDerivativeType* gradient,
  const unsigned int pos) const
{

  /** Check if mapped point inside image buffer. */
  MovingImageContinuousIndexType cindex;
  this->m_InterpolatorVector[pos]->ConvertPointToContinuousIndex(mappedPoint, cindex);
  bool sampleOk = this->GetInterpolator(pos)->IsInsideBuffer(cindex);
  if (sampleOk)
  {
    /** Compute value and possibly derivative. */
    if (gradient)
    {
      if (this->m_InterpolatorsAreBSpline && !this->GetComputeGradient())
      {
        /** Compute moving image value and gradient using the B-spline kernel. */
        this->GetBSplineInterpolator(pos)->EvaluateValueAndDerivativeAtContinuousIndex(
          cindex, movingImageValue, *gradient);
      }
      else
      {
        /** Get the gradient by NearestNeighboorInterpolation of the gradient image.
         * It is assumed that the gradient image is computed.
         */
        movingImageValue = this->GetInterpolator(pos)->EvaluateAtContinuousIndex(cindex);
        MovingImageIndexType index;
        for (unsigned int j = 0; j < MovingImageDimension; j++)
        {
          index[j] = static_cast<long>(Math::Round< double >(cindex[j]));
        }
        (*gradient) = this->m_GradientImage->GetPixel(index);
      }
    } // end if gradient
    else
    {
      movingImageValue = this->GetInterpolator(pos)->EvaluateAtContinuousIndex(cindex);
    }
  } // end if sampleOk

  return sampleOk;

} // end EvaluateMovingImageValueAndDerivative()





/**
 * ************************ IsInsideMovingMask *************************
 */

template< class TFixedImage, class TMovingImage >
bool
MultiInputImageToImageMetricBase< TFixedImage, TMovingImage >
::IsInsideMovingMask( const MovingImagePointType & mappedPoint ) const
{
  /** If no moving image masks are present 'true' is returned,
   * meaning that this sample is taken into account. Otherwise, the
   * AND of all masks is returned, i.e. the sample should be inside
   * all masks.
   */
  bool inside = true;
  for( unsigned int i = 0; i < this->GetNumberOfMovingImageMasks(); ++i )
  {
    MovingImageMaskPointer movingImageMask = this->GetMovingImageMask( i );
    if( movingImageMask.IsNotNull() )
    {
      inside &= movingImageMask->IsInsideInWorldSpace( mappedPoint );
    }

    /** If the point falls outside one mask, we can skip the rest. */
    if( !inside )
    {
      return false;
    }
  }
  return inside;

} // end IsInsideMovingMask()


} // end namespace itk

#undef itkImplementationSetObjectMacro
#undef itkImplementationSetObjectMacro2
#undef itkImplementationGetObjectMacro
#undef itkImplementationGetConstObjectMacro

#endif // end #ifndef _itkMultiInputImageToImageMetricBase_hxx
