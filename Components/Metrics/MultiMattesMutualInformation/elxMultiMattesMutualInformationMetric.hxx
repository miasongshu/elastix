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
#ifndef __elxMultiMattesMutualInformationMetric_HXX__
#define __elxMultiMattesMutualInformationMetric_HXX__

#include "elxMultiMattesMutualInformationMetric.h"

#include "itkHardLimiterFunction.h"
#include "itkExponentialLimiterFunction.h"
#include <string>
#include "vnl/vnl_math.h"
#include "itkTimeProbe.h"

namespace elastix
{

/**
 * ****************** Constructor ***********************
 */

template< class TElastix >
MultiMattesMutualInformationMetric< TElastix >
::MultiMattesMutualInformationMetric()
{
#ifdef BENCHMARK  /* Just for TESTING analytic derivative vs. numeric (finite difference) */
  this->m_CurrentIteration = 0.0;
  this->m_Param_c          = 1.0;
  this->m_Param_gamma      = 0.101;
#endif
  this->SetUseDerivative(true);
} // end Constructor()


/**
 * ******************* Initialize ***********************
 */

template< class TElastix >
void
MultiMattesMutualInformationMetric< TElastix >
::Initialize( void )
{
  itk::TimeProbe timer;
  timer.Start();
  this->Superclass1::Initialize();
  timer.Stop();
  elxout << "Initialization of MultiMattesMutualInformation metric took: "
         << static_cast< long >( timer.GetMean() * 1000 ) << " ms." << std::endl;

} // end Initialize()


/**
 * ***************** BeforeEachResolution ***********************
 */

template< class TElastix >
void
MultiMattesMutualInformationMetric< TElastix >
::BeforeEachResolution( void )
{
  /** Get the current resolution level. */
  unsigned int level
    = ( this->m_Registration->GetAsITKBaseType() )->GetCurrentLevel();

  /** Get and set the number of histogram bins. */
  unsigned int numberOfHistogramBins = 32;
  this->GetConfiguration()->ReadParameter( numberOfHistogramBins,
    "NumberOfHistogramBins", this->GetComponentLabel(), level, 0 );
  this->SetNumberOfFixedHistogramBins( numberOfHistogramBins );
  this->SetNumberOfMovingHistogramBins( numberOfHistogramBins );

  unsigned int numberOfFixedHistogramBins  = numberOfHistogramBins;
  unsigned int numberOfMovingHistogramBins = numberOfHistogramBins;
  this->GetConfiguration()->ReadParameter( numberOfFixedHistogramBins,
    "NumberOfFixedHistogramBins", this->GetComponentLabel(), level, 0 );
  this->GetConfiguration()->ReadParameter( numberOfMovingHistogramBins,
    "NumberOfMovingHistogramBins", this->GetComponentLabel(), level, 0 );
  this->SetNumberOfFixedHistogramBins( numberOfFixedHistogramBins );
  this->SetNumberOfMovingHistogramBins( numberOfMovingHistogramBins );

  // There is only one set here - the rest happens in InitializeLimiter

  bool useFixedImageLimiter = true;
  this->GetConfiguration()->ReadParameter(useFixedImageLimiter,
    "useFixedImageLimiter", this->GetComponentLabel(), level, 0);
  this->SetUseFixedImageLimiter(useFixedImageLimiter);

  bool useMovingImageLimiter = true;
  this->GetConfiguration()->ReadParameter(useMovingImageLimiter,
    "useMovingImageLimiter", this->GetComponentLabel(), level, 0);
  this->SetUseMovingImageLimiter(useMovingImageLimiter);

  ///** Set limiters. */
  //typedef itk::HardLimiterFunction< RealType, FixedImageDimension >         FixedLimiterType;
  //typedef itk::ExponentialLimiterFunction< RealType, MovingImageDimension > MovingLimiterType;
  //this->SetFixedImageLimiter( FixedLimiterType::New() );
  //this->SetMovingImageLimiter( MovingLimiterType::New() );

  /** Get and set the limit range ratios. */
  double fixedLimitRangeRatio  = 0.01;
  double movingLimitRangeRatio = 0.01;
  this->GetConfiguration()->ReadParameter( fixedLimitRangeRatio,
    "FixedLimitRangeRatio", this->GetComponentLabel(), level, 0 );
  this->GetConfiguration()->ReadParameter( movingLimitRangeRatio,
    "MovingLimitRangeRatio", this->GetComponentLabel(), level, 0 );
  this->SetFixedLimitRangeRatio( fixedLimitRangeRatio );
  this->SetMovingLimitRangeRatio( movingLimitRangeRatio );

  /** Set B-spline Parzen kernel orders. */
  unsigned int fixedKernelBSplineOrder  = 0;
  unsigned int movingKernelBSplineOrder = 3;
  this->GetConfiguration()->ReadParameter( fixedKernelBSplineOrder,
    "FixedKernelBSplineOrder", this->GetComponentLabel(), level, 0 );
  this->GetConfiguration()->ReadParameter( movingKernelBSplineOrder,
    "MovingKernelBSplineOrder", this->GetComponentLabel(), level, 0 );
  this->SetFixedKernelBSplineOrder( fixedKernelBSplineOrder );
  this->SetMovingKernelBSplineOrder( movingKernelBSplineOrder );


#ifdef BENCHMARK  /* Just for TESTING analytic derivative vs. numeric (finite difference) */
  /** Prepare for computing the perturbation gain c_k. */
  this->SetCurrentIteration( 0 );

  {
    double c = 1.0;
    double gamma = 0.101;
    this->GetConfiguration()->ReadParameter(c, "SP_c",
      this->GetComponentLabel(), level, 0);
    this->GetConfiguration()->ReadParameter(gamma, "SP_gamma",
      this->GetComponentLabel(), level, 0);
    this->SetParam_c(c);
    this->SetParam_gamma(gamma);
    this->SetFiniteDifferencePerturbation(this->Compute_c(0));
  }
#endif

} // end BeforeEachResolution()


#ifdef BENCHMARK  /* Just for TESTING analytic derivative vs. numeric (finite difference) */
/**
 * ***************** AfterEachIteration ***********************
 */

template< class TElastix >
void
MultiMattesMutualInformationMetric< TElastix >
::AfterEachIteration(void)
{
    this->m_CurrentIteration++;
    this->SetFiniteDifferencePerturbation(
      this->Compute_c(this->m_CurrentIteration));
} // end AfterEachIteration()


/**
 * ************************** Compute_c *************************
 */
template< class TElastix >
double
MultiMattesMutualInformationMetric< TElastix >
::Compute_c( unsigned long k ) const
{
  return static_cast< double >(
    this->m_Param_c / std::pow( k + 1, this->m_Param_gamma ) );

} // end Compute_c()
#endif

} // end namespace elastix

#endif // end #ifndef __elxMultiMattesMutualInformationMetric_HXX__
