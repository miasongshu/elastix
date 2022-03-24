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
#ifndef _itkMultiPWHistogramImageToImageMetric_HXX__
#define _itkMultiPWHistogramImageToImageMetric_HXX__

#include "itkMultiPWHistogramImageToImageMetric.h"

#include "itkBSplineKernelFunction2.h"
#include "itkBSplineDerivativeKernelFunction2.h"
#include "itkImageLinearIteratorWithIndex.h"
#include "itkImageScanlineIterator.h"
#include "vnl/vnl_math.h"

namespace itk
{

/**
 * ********************* Constructor ****************************
 */

template< class TFixedImage, class TMovingImage >
MultiPWHistogramImageToImageMetric< TFixedImage, TMovingImage >
::MultiPWHistogramImageToImageMetric()
{
  this->m_NumberOfFixedHistogramBins        = 32;
  this->m_NumberOfMovingHistogramBins       = 32;
  this->m_JointPDFVector                    = {};
  this->m_JointPDFDerivativesVector         = {};
  this->m_FixedImageNormalizedMinVector     = {};
  this->m_MovingImageNormalizedMinVector    = {};
  this->m_FixedImageBinSizeVector           = {};
  this->m_MovingImageBinSizeVector          = {};
  this->m_AlphaVector                       = {};
  this->m_FixedImageMarginalPDFVector       = {};
  this->m_MovingImageMarginalPDFVector      = {};

  this->m_FixedKernel                   = 0;
  this->m_MovingKernel                  = 0;
  this->m_DerivativeMovingKernel        = 0;
  this->m_FixedKernelBSplineOrder       = 0;
  this->m_MovingKernelBSplineOrder      = 3;
  this->m_FixedParzenTermToIndexOffset  = 0.5;
  this->m_MovingParzenTermToIndexOffset = -1.0;

  this->SetUseImageSampler( true );
  this->SetUseFixedImageLimiter( true );
  this->SetUseMovingImageLimiter( true );


} // end Constructor




/**
 * ********************* Initialize *****************************
 */

template< class TFixedImage, class TMovingImage >
void
MultiPWHistogramImageToImageMetric< TFixedImage, TMovingImage >
::Initialize( void )
{
  /** Call the superclass to check that standard components are available. */
  this->Superclass::Initialize();

  /** Set up the histograms. */
  this->InitializeHistograms();

  /** Set up the Parzen windows. */
  this->InitializeKernels();


} // end Initialize()


/**
 * ****************** InitializeHistograms *****************************
 */

template< class TFixedImage, class TMovingImage >
void
MultiPWHistogramImageToImageMetric< TFixedImage, TMovingImage >
::InitializeHistograms(void)
{
  /* Compute binsize for the histogram.
   *
   * The binsize for the image intensities needs to be adjusted so that
   * we can avoid dealing with boundary conditions using the cubic
   * spline as the Parzen window.  We do this by increasing the size
   * of the bins so that the joint histogram becomes "padded" at the
   * borders. Because we are changing the binsize,
   * we also need to shift the minimum by the padded amount in order to
   * avoid minimum values filling in our padded region.
   *
   * Note that there can still be non-zero bin values in the padded region,
   * it's just that these bins will never be a central bin for the Parzen
   * window.
   */
   //int fixedPadding = 2;  // this will pad by 2 bins
   //int movingPadding = 2;  // this will pad by 2 bins
  int fixedPadding = this->m_FixedKernelBSplineOrder / 2; // should be enough
  int movingPadding = this->m_MovingKernelBSplineOrder / 2;


  const unsigned int clusterSize = this->GetNumberOfFixedImages();
  /** Loop over all the multiple images in the cluster */
  for (unsigned int pos = 0; pos < clusterSize; ++pos)
  {
    /** The ratio times the expected bin size will be added twice to the image range. */
    const double smallNumberRatio = 0.001;
    const double smallNumberFixed = smallNumberRatio
    * (this->m_FixedImageMaxLimitVector[pos] - this->m_FixedImageMinLimitVector[pos])
    / static_cast<double>(this->m_NumberOfFixedHistogramBins - 2 * fixedPadding - 1);
  const double smallNumberMoving = smallNumberRatio
    * (this->m_MovingImageMaxLimitVector[pos] - this->m_MovingImageMinLimitVector[pos])
    / static_cast<double>(this->m_NumberOfFixedHistogramBins - 2 * movingPadding - 1);

  /** Compute binsizes. */
  const double fixedHistogramWidth = static_cast<double>(
    static_cast<OffsetValueType>(this->m_NumberOfFixedHistogramBins) // requires cast to signed type!
    - 2.0 * fixedPadding - 1.0);
  this->m_FixedImageBinSizeVector.push_back(
    (this->m_FixedImageMaxLimitVector[pos] - this->m_FixedImageMinLimitVector[pos]
      + 2.0 * smallNumberFixed) / fixedHistogramWidth);
  this->m_FixedImageBinSizeVector[pos] = std::max(this->m_FixedImageBinSizeVector[pos], 1e-10);
  this->m_FixedImageBinSizeVector[pos] = std::min(this->m_FixedImageBinSizeVector[pos], 1e+10);
  this->m_FixedImageNormalizedMinVector.push_back(
    (this->m_FixedImageMinLimitVector[pos] - smallNumberFixed)
    / this->m_FixedImageBinSizeVector[pos] - static_cast<double>(fixedPadding));

  const double movingHistogramWidth = static_cast<double>(
    static_cast<OffsetValueType>(this->m_NumberOfMovingHistogramBins) // requires cast to signed type!
    - 2.0 * movingPadding - 1.0);
  this->m_MovingImageBinSizeVector.push_back(
    (this->m_MovingImageMaxLimitVector[pos] - this->m_MovingImageMinLimitVector[pos]
      + 2.0 * smallNumberMoving) / movingHistogramWidth);
  this->m_MovingImageBinSizeVector[pos] = std::max(this->m_MovingImageBinSizeVector[pos], 1e-10);
  this->m_MovingImageBinSizeVector[pos] = std::min(this->m_MovingImageBinSizeVector[pos], 1e+10);
  this->m_MovingImageNormalizedMinVector.push_back(
    (this->m_MovingImageMinLimitVector[pos] - smallNumberMoving)
    / this->m_MovingImageBinSizeVector[pos] - static_cast<double>(movingPadding));

  /** Allocate memory for the marginal PDF. */
  MarginalPDFType FixedImageMarginalPDFDummy = {};
  FixedImageMarginalPDFDummy.SetSize(this->m_NumberOfFixedHistogramBins);
  this->m_FixedImageMarginalPDFVector.push_back(FixedImageMarginalPDFDummy);
  MarginalPDFType MovingImageMarginalPDFDummy = {};
  MovingImageMarginalPDFDummy.SetSize(this->m_NumberOfMovingHistogramBins);
  this->m_MovingImageMarginalPDFVector.push_back(MovingImageMarginalPDFDummy);

  //this->m_FixedImageMarginalPDFVector.SetSize(this->m_NumberOfFixedHistogramBins);
  //this->m_MovingImageMarginalPDFVector.SetSize(this->m_NumberOfMovingHistogramBins);

  /** Allocate memory for the joint PDF and joint PDF derivatives. */

  /** For the joint PDF define a region starting from {0,0}
   * with size {this->m_NumberOfMovingHistogramBins, this->m_NumberOfFixedHistogramBins}
   * The dimension represents moving image Parzen window index
   * and fixed image Parzen window index, respectively.
   * The moving Parzen index is chosen as the first dimension,
   * because probably the moving B-spline kernel order will be larger
   * than the fixed B-spline kernel order and it is faster to iterate along
   * the first dimension.
   */
  this->m_JointPDFVector.push_back(JointPDFType::New());
  JointPDFRegionType jointPDFRegion;
  JointPDFIndexType  jointPDFIndex;
  JointPDFSizeType   jointPDFSize;
  jointPDFIndex.Fill(0);
  jointPDFSize[0] = this->m_NumberOfMovingHistogramBins;
  jointPDFSize[1] = this->m_NumberOfFixedHistogramBins;
  jointPDFRegion.SetIndex(jointPDFIndex);
  jointPDFRegion.SetSize(jointPDFSize);
  this->m_JointPDFVector[pos]->SetRegions(jointPDFRegion);
  this->m_JointPDFVector[pos]->Allocate();

  this->m_AlphaVector.push_back(0);

  if (this->GetUseDerivative())
  {
    /** For the derivatives of the joint PDF define a region starting from {0,0,0}
     * with size {GetNumberOfParameters(),m_NumberOfMovingHistogramBins,
     * m_NumberOfFixedHistogramBins}. The dimension represents transform parameters,
     * moving image Parzen window index and fixed image Parzen window index,
     * respectively.
     */

    JointPDFDerivativesRegionType jointPDFDerivativesRegion;
    JointPDFDerivativesIndexType  jointPDFDerivativesIndex;
    JointPDFDerivativesSizeType   jointPDFDerivativesSize;
    jointPDFDerivativesIndex.Fill(0);
    jointPDFDerivativesSize[0] = this->GetNumberOfParameters();
    jointPDFDerivativesSize[1] = this->m_NumberOfMovingHistogramBins;
    jointPDFDerivativesSize[2] = this->m_NumberOfFixedHistogramBins;
    jointPDFDerivativesRegion.SetIndex(jointPDFDerivativesIndex);
    jointPDFDerivativesRegion.SetSize(jointPDFDerivativesSize);


    this->m_JointPDFDerivativesVector.push_back(JointPDFDerivativesType::New());
    this->m_JointPDFDerivativesVector[pos]->SetRegions(jointPDFDerivativesRegion);
    this->m_JointPDFDerivativesVector[pos]->Allocate();
  }
  else
  {
    this->m_JointPDFDerivativesVector.push_back(nullptr);
  }
}// end loop over pos
} // end InitializeHistograms()


/**
 * ****************** InitializeKernels *****************************
 */

template< class TFixedImage, class TMovingImage >
void
MultiPWHistogramImageToImageMetric< TFixedImage, TMovingImage >
::InitializeKernels( void )
{
  switch( this->m_FixedKernelBSplineOrder )
  {
    case 0:
      this->m_FixedKernel = BSplineKernelFunction2< 0 >::New(); break;
    case 1:
      this->m_FixedKernel = BSplineKernelFunction2< 1 >::New(); break;
    case 2:
      this->m_FixedKernel = BSplineKernelFunction2< 2 >::New(); break;
    case 3:
      this->m_FixedKernel = BSplineKernelFunction2< 3 >::New(); break;
    default:
      itkExceptionMacro( << "The following FixedKernelBSplineOrder is not implemented: " \
                         << this->m_FixedKernelBSplineOrder );
  } // end switch FixedKernelBSplineOrder

  switch( this->m_MovingKernelBSplineOrder )
  {
    case 0:
      this->m_MovingKernel = BSplineKernelFunction2< 0 >::New();
      /** The derivative of a zero order B-spline makes no sense. Using the
       * derivative of a first order gives a kind of finite difference idea
       * Anyway, if you plan to call GetValueAndDerivative you should use
       * a higher B-spline order.
       */
      this->m_DerivativeMovingKernel = BSplineDerivativeKernelFunction2< 1 >::New();
      break;
    case 1:
      this->m_MovingKernel           = BSplineKernelFunction2< 1 >::New();
      this->m_DerivativeMovingKernel = BSplineDerivativeKernelFunction2< 1 >::New();
      break;
    case 2:
      this->m_MovingKernel           = BSplineKernelFunction2< 2 >::New();
      this->m_DerivativeMovingKernel = BSplineDerivativeKernelFunction2< 2 >::New();
      break;
    case 3:
      this->m_MovingKernel           = BSplineKernelFunction2< 3 >::New();
      this->m_DerivativeMovingKernel = BSplineDerivativeKernelFunction2< 3 >::New();
      break;
    default:
      itkExceptionMacro( << "The following MovingKernelBSplineOrder is not implemented: " \
                         << this->m_MovingKernelBSplineOrder );
  } // end switch MovingKernelBSplineOrder

  /** The region of support of the Parzen window determines which bins
   * of the joint PDF are effected by the pair of image values.
   * For example, if we are using a cubic spline for the moving image Parzen
   * window, four bins are affected. If the fixed image Parzen window is
   * a zero-order spline (box car) only one bin is affected.
   */

  /** Set the size of the Parzen window. */
  JointPDFSizeType parzenWindowSize;
  parzenWindowSize[ 0 ] = this->m_MovingKernelBSplineOrder + 1;
  parzenWindowSize[ 1 ] = this->m_FixedKernelBSplineOrder + 1;

  this->m_JointPDFWindow.SetSize( parzenWindowSize );

  /** The ParzenIndex is the lowest bin number that is affected by a
   * pixel and computed as:
   * ParzenIndex = std::floor( ParzenTerm + ParzenTermToIndexOffset )
   * where ParzenTermToIndexOffset = 1/2, 0, -1/2, or -1.
   */
  this->m_FixedParzenTermToIndexOffset
    = 0.5 - static_cast< double >( this->m_FixedKernelBSplineOrder ) / 2.0;
  this->m_MovingParzenTermToIndexOffset
    = 0.5 - static_cast< double >( this->m_MovingKernelBSplineOrder ) / 2.0;

} // end InitializeKernels()



/**
 * ******************** GetDerivative ***************************
 */

template< class TFixedImage, class TMovingImage >
void
MultiPWHistogramImageToImageMetric< TFixedImage, TMovingImage >
::GetDerivative( const ParametersType & parameters, DerivativeType & derivative ) const
{
  /** Call the combined version, since the additional computation of
   * the value does not take extra time.
   */
  MeasureType value;
  this->GetValueAndDerivative( parameters, value, derivative );

} // end GetDerivative()


/**
 * ******************** GetValueAndDerivative ***************************
 */

template< class TFixedImage, class TMovingImage >
void
MultiPWHistogramImageToImageMetric< TFixedImage, TMovingImage >
::GetValueAndDerivative( const ParametersType & parameters,
  MeasureType & value, DerivativeType & derivative ) const
{
   this->GetValueAndAnalyticDerivative( parameters, value, derivative );
} // end GetValueAndDerivative()


/**
 * ********************** EvaluateParzenValues ***************
 */

template< class TFixedImage, class TMovingImage >
void
MultiPWHistogramImageToImageMetric< TFixedImage, TMovingImage >
::EvaluateParzenValues(
  double parzenWindowTerm, OffsetValueType parzenWindowIndex,
  const KernelFunctionType * kernel, ParzenValueContainerType & parzenValues ) const
{
  kernel->Evaluate( static_cast<double>( parzenWindowIndex ) - parzenWindowTerm, parzenValues.data_block() );
} // end EvaluateParzenValues()


/**
 * ********************** UpdateJointPDFAndDerivatives ***************
 */

template< class TFixedImage, class TMovingImage >
void
MultiPWHistogramImageToImageMetric< TFixedImage, TMovingImage >
::UpdateJointPDFAndDerivatives(
  const RealType & fixedImageValue,
  const RealType & movingImageValue,
  const DerivativeType * imageJacobian,
  const NonZeroJacobianIndicesType * nzji,
  JointPDFType * jointPDF,
  const unsigned int pos) const
{
  typedef ImageScanlineIterator< JointPDFType > PDFIteratorType;

  /** Determine Parzen window arguments (see eq. 6 of Mattes paper [2]). */
  const double fixedImageParzenWindowTerm
    = fixedImageValue / this->m_FixedImageBinSizeVector[pos] - this->m_FixedImageNormalizedMinVector[pos];
  const double movingImageParzenWindowTerm
    = movingImageValue / this->m_MovingImageBinSizeVector[pos] - this->m_MovingImageNormalizedMinVector[pos] ;

  /** The lowest bin numbers affected by this pixel: */
  const OffsetValueType fixedImageParzenWindowIndex
    = static_cast< OffsetValueType >( std::floor(
    fixedImageParzenWindowTerm + this->m_FixedParzenTermToIndexOffset ) );
  const OffsetValueType movingImageParzenWindowIndex
    = static_cast< OffsetValueType >( std::floor(
    movingImageParzenWindowTerm + this->m_MovingParzenTermToIndexOffset ) );

  /** The Parzen values. */
  ParzenValueContainerType fixedParzenValues( this->m_JointPDFWindow.GetSize()[ 1 ] );
  ParzenValueContainerType movingParzenValues( this->m_JointPDFWindow.GetSize()[ 0 ] );
  this->EvaluateParzenValues(
    fixedImageParzenWindowTerm, fixedImageParzenWindowIndex,
    this->m_FixedKernel, fixedParzenValues );
  this->EvaluateParzenValues(
    movingImageParzenWindowTerm, movingImageParzenWindowIndex,
    this->m_MovingKernel, movingParzenValues );

  /** Position the JointPDFWindow. */
  JointPDFIndexType pdfWindowIndex;
  pdfWindowIndex[ 0 ] = movingImageParzenWindowIndex;
  pdfWindowIndex[ 1 ] = fixedImageParzenWindowIndex;

  /** For thread-safety, make a local copy of the support region,
   * and use that one. Because each thread will modify it.
   */
  JointPDFRegionType jointPDFWindow = this->m_JointPDFWindow;
  jointPDFWindow.SetIndex( pdfWindowIndex );
  PDFIteratorType it( jointPDF, jointPDFWindow );

  if( !imageJacobian )
  {
    /** Loop over the Parzen window region and increment the values. */
    for( unsigned int f = 0; f < fixedParzenValues.GetSize(); ++f )
    {
      const double fv = fixedParzenValues[ f ];
      for( unsigned int m = 0; m < movingParzenValues.GetSize(); ++m )
      {
        it.Value() += static_cast< PDFValueType >( fv * movingParzenValues[ m ] );
        ++it;
      }
      it.NextLine();
    }
  }
  else
  {
    /** Compute the derivatives of the moving Parzen window. */
    ParzenValueContainerType derivativeMovingParzenValues(
    this->m_JointPDFWindow.GetSize()[ 0 ] );
    this->EvaluateParzenValues(
      movingImageParzenWindowTerm, movingImageParzenWindowIndex,
      this->m_DerivativeMovingKernel, derivativeMovingParzenValues );

    const double et = static_cast< double >( this->m_MovingImageBinSizeVector[pos]);

    /** Loop over the Parzen window region and increment the values
     * Also update the pdf derivatives.
     */
    for( unsigned int f = 0; f < fixedParzenValues.GetSize(); ++f )
    {
      const double fv    = fixedParzenValues[ f ];
      const double fv_et = fv / et;
      for( unsigned int m = 0; m < movingParzenValues.GetSize(); ++m )
      {
        it.Value() += static_cast< PDFValueType >( fv * movingParzenValues[ m ] );
        this->UpdateJointPDFDerivatives(
          it.GetIndex(), fv_et * derivativeMovingParzenValues[ m ],
          *imageJacobian, *nzji, pos );
        ++it;
      }
      it.NextLine();
    }
  }

} // end UpdateJointPDFAndDerivatives()


/**
 * *************** UpdateJointPDFDerivatives ***************************
 */

template< class TFixedImage, class TMovingImage >
void
MultiPWHistogramImageToImageMetric< TFixedImage, TMovingImage >
::UpdateJointPDFDerivatives(
  const JointPDFIndexType & pdfIndex, double factor,
  const DerivativeType & imageJacobian,
  const NonZeroJacobianIndicesType & nzji,
  const unsigned int pos) const
{
  /** Get the pointer to the element with index [0, pdfIndex[0], pdfIndex[1]]. */
  PDFDerivativeValueType * derivPtr = this->m_JointPDFDerivativesVector[pos]->GetBufferPointer()
    + ( pdfIndex[ 0 ] * this->m_JointPDFDerivativesVector[pos]->GetOffsetTable()[ 1 ] )
    + ( pdfIndex[ 1 ] * this->m_JointPDFDerivativesVector[pos]->GetOffsetTable()[ 2 ] );

  if( nzji.size() == this->GetNumberOfParameters() )
  {
    /** Loop over all Jacobians. */
    typename DerivativeType::const_iterator imjac = imageJacobian.begin();
    for( unsigned int mu = 0; mu < this->GetNumberOfParameters(); ++mu )
    {
      *( derivPtr ) -= static_cast< PDFDerivativeValueType >( ( *imjac ) * factor );
      ++derivPtr;
      ++imjac;
    }
  }
  else
  {
    /** Loop only over the non-zero Jacobians. */
    for( unsigned int i = 0; i < imageJacobian.GetSize(); ++i )
    {
      const unsigned int       mu  = nzji[ i ];
      PDFDerivativeValueType * ptr = derivPtr + mu;
      *( ptr ) -= static_cast< PDFDerivativeValueType >( imageJacobian[ i ] * factor );
    }
  }

} // end UpdateJointPDFDerivatives()


/**
 * *********************** NormalizeJointPDF ***********************
 */

template< class TFixedImage, class TMovingImage >
void
MultiPWHistogramImageToImageMetric< TFixedImage, TMovingImage >
::NormalizeJointPDF( JointPDFType * pdf, const double & factor ) const
{
  typedef ImageScanlineIterator< JointPDFType > JointPDFIteratorType;
  JointPDFIteratorType it( pdf, pdf->GetBufferedRegion() );
  const PDFValueType   castfac = static_cast< PDFValueType >( factor );
  while( !it.IsAtEnd() )
  {
    while( !it.IsAtEndOfLine() )
    {
      it.Value() *= castfac;
      ++it;
    }
    it.NextLine();
  }

} // end NormalizeJointPDF()


/**
 * *********************** NormalizeJointPDFDerivatives ***********************
 */

template< class TFixedImage, class TMovingImage >
void
MultiPWHistogramImageToImageMetric< TFixedImage, TMovingImage >
::NormalizeJointPDFDerivatives( JointPDFDerivativesType * pdf, const double & factor ) const
{
  typedef ImageScanlineIterator< JointPDFDerivativesType > JointPDFDerivativesIteratorType;
  JointPDFDerivativesIteratorType it( pdf, pdf->GetBufferedRegion() );
  const PDFValueType              castfac = static_cast< PDFValueType >( factor );
  while( !it.IsAtEnd() )
  {
    while( !it.IsAtEndOfLine() )
    {
      it.Value() *= castfac;
      ++it;
    }
    it.NextLine();
  }

} // end NormalizeJointPDFDerivatives()

/**
 * ************************ ComputeMarginalPDF ***********************
 */

template< class TFixedImage, class TMovingImage >
void
MultiPWHistogramImageToImageMetric< TFixedImage, TMovingImage >
::ComputeMarginalPDF(
  const JointPDFType * jointPDF,
  MarginalPDFType & marginalPDF, const unsigned int & direction, const unsigned int pos ) const
{
  typedef ImageLinearIteratorWithIndex< JointPDFType > JointPDFLinearIterator;
  // \todo: bug? shouldn't this be over the function argument jointPDF ?
  JointPDFLinearIterator linearIter(this->m_JointPDFVector[pos], this->m_JointPDFVector[pos]->GetBufferedRegion());
  //JointPDFLinearIterator linearIter( jointPDF, jointPDF->GetBufferedRegion() ); // not possible??? Songshu
  linearIter.SetDirection( direction );
  linearIter.GoToBegin();
  unsigned int marginalIndex = 0;
  while( !linearIter.IsAtEnd() )
  {
    PDFValueType sum = 0.0;
    while( !linearIter.IsAtEndOfLine() )
    {
      sum += linearIter.Get();
      ++linearIter;
    }
    marginalPDF[ marginalIndex ] = sum;
    linearIter.NextLine();
    ++marginalIndex;
  }

} // end ComputeMarginalPDFs()





/**
 * ************************ ComputePDFs **************************
 */

template< class TFixedImage, class TMovingImage >
void
MultiPWHistogramImageToImageMetric< TFixedImage, TMovingImage >
::ComputePDFs( const ParametersType & parameters, const unsigned int pos ) const
{
  /** Initialize some variables. */
  this->m_JointPDFVector[pos]->FillBuffer(0.0);
  this->m_NumberOfPixelsCounted = 0;
  this->m_AlphaVector[pos] = 0.0;  // not possible??? Songshu


  /** Get a handle to the sample container. */
  ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();

  /** Create iterator over the sample container. */
  typename ImageSampleContainerType::ConstIterator fiter;
  typename ImageSampleContainerType::ConstIterator fbegin = sampleContainer->Begin();
  typename ImageSampleContainerType::ConstIterator fend = sampleContainer->End();

  /** Loop over sample container and compute contribution of each sample to pdfs. */
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
     * inside the moving image buffer.
     */
    if (sampleOk)
    {
      sampleOk = this->EvaluateMovingImageValueAndDerivative(
        mappedPoint, movingImageValue, 0, pos);
    }

    if (sampleOk)
    {
      this->m_NumberOfPixelsCounted++;

      /** Get the fixed image value. */
      RealType fixedImageValue = static_cast<RealType>((*fiter).Value().m_ImageValue);

      /** Make sure the values fall within the histogram range. */
      fixedImageValue = this->GetFixedImageLimiter(pos)->Evaluate(fixedImageValue);
      movingImageValue = this->GetMovingImageLimiter(pos)->Evaluate(movingImageValue);

      /** Compute this sample's contribution to the joint distributions. */
      this->UpdateJointPDFAndDerivatives(
        fixedImageValue, movingImageValue, 0, 0, this->m_JointPDFVector[pos].GetPointer(), pos);  // works automatically (?)
    }

  } // end iterating over fixed image spatial sample container for loop

  /** Check if enough samples were valid. */
  this->CheckNumberOfSamples(sampleContainer->Size(), this->m_NumberOfPixelsCounted);

  /** Compute alpha. */
  this->m_AlphaVector[pos] = 1.0 / static_cast<double>(this->m_NumberOfPixelsCounted);

} // end ComputePDFs()




/**
 * ************************ ComputePDFsAndPDFDerivatives *******************
 */

template< class TFixedImage, class TMovingImage >
void
MultiPWHistogramImageToImageMetric< TFixedImage, TMovingImage >
::ComputePDFsAndPDFDerivatives( const ParametersType & parameters, unsigned int pos) const
{
  /** Initialize some variables. */
  this->m_JointPDFVector[pos]->FillBuffer( 0.0 );
  this->m_JointPDFDerivativesVector[pos]->FillBuffer( 0.0 );
  this->m_AlphaVector[pos] = 0.0;  // not possible??? Songshu
  this->m_NumberOfPixelsCounted = 0;
  /** Array that stores dM(x)/dmu, and the sparse jacobian+indices. */
  NonZeroJacobianIndicesType nzji( this->m_AdvancedTransform->GetNumberOfNonZeroJacobianIndices() );
  DerivativeType             imageJacobian( nzji.size() );
  TransformJacobianType      jacobian;


    /** Get a handle to the sample container. */
    ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();
    /** Create iterator over the sample container. */
    typename ImageSampleContainerType::ConstIterator fiter;
    typename ImageSampleContainerType::ConstIterator fbegin = sampleContainer->Begin();
    typename ImageSampleContainerType::ConstIterator fend = sampleContainer->End();

    /** Loop over sample container and compute contribution of each sample to pdfs. */
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
        mappedPoint, movingImageValue, &movingImageDerivative, pos );
      }
      if (sampleOk)
      {
        this->m_NumberOfPixelsCounted++;

        /** Get the fixed image value. */
        RealType fixedImageValue = static_cast<RealType>((*fiter).Value().m_ImageValue);

        /** Make sure the values fall within the histogram range. */
        fixedImageValue = this->GetFixedImageLimiter(pos)->Evaluate(fixedImageValue);
        movingImageValue = this->GetMovingImageLimiter(pos)->Evaluate(
          movingImageValue, movingImageDerivative);
        /** Get the TransformJacobian dT/dmu. */
        this->EvaluateTransformJacobian(fixedPoint, jacobian, nzji);
        /** Compute the inner product (dM/dx)^T (dT/dmu). */
        this->EvaluateTransformJacobianInnerProduct(
          jacobian, movingImageDerivative, imageJacobian);
        /** Update the joint pdf and the joint pdf derivatives. */
        this->UpdateJointPDFAndDerivatives(
          fixedImageValue, movingImageValue, &imageJacobian, &nzji, this->m_JointPDFVector[pos].GetPointer(), pos);

      } //end if-block check sampleOk
    } // end iterating over fixed image spatial sample container for loop
   
    /** Check if enough samples were valid. */
    this->CheckNumberOfSamples(
      sampleContainer->Size(), this->m_NumberOfPixelsCounted);

    /** Compute alpha. */
  this->m_AlphaVector[pos] = 0.0;
    if (this->m_NumberOfPixelsCounted > 0)
    {
    this->m_AlphaVector[pos] = 1.0 / static_cast< double >( this->m_NumberOfPixelsCounted );
    }
} // end ComputePDFsAndPDFDerivatives()


} // end namespace itk

#endif // end #ifndef _itkMultiPWHistogramImageToImageMetric_HXX__
