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
#ifndef ELXAFFINELOGTRANSFORMOLASTDIM_HXX
#define ELXAFFINELOGTRANSFORMOLASTDIM_HXX

#include "elxAffineLogTransformOLastDim.h"
#include "itkContinuousIndex.h"

namespace elastix
{

/**
 * ********************* Constructor ****************************
 */

template< class TElastix >
AffineLogTransformOLastDimElastix< TElastix >::AffineLogTransformOLastDimElastix()
{} // end Constructor



/**
* ********************* InitializeAffineTransform ****************************
*/
template< class TElastix >
unsigned int
AffineLogTransformOLastDimElastix< TElastix >
::InitializeAffineLogTransform()
{
  this->m_AffineLogTransformOLastDim = ReducedDimensionAffineLogTransformBaseType::New();
  this->SetCurrentTransform(this->m_AffineLogTransformOLastDim);

  return 0;
}


/**
 * ******************* BeforeAll ***********************
 */

template< class TElastix >
int
AffineLogTransformOLastDimElastix< TElastix >
::BeforeAll(void)
{
  /** Initialize affine transform. */
  return InitializeAffineLogTransform();
}


/**
 * ******************* BeforeRegistration ***********************
 */

template< class TElastix >
void
AffineLogTransformOLastDimElastix< TElastix >
::BeforeRegistration( void )
{
  elxout << "BeforeRegistration" << std::endl;
  /** Set center of rotation and initial translation. */
  this->InitializeTransform();

  /** Set the scales. */
  this->SetScales();

} // end BeforeRegistration()


/**
 * ************************* ReadFromFile ************************
 */

template< class TElastix >
void
AffineLogTransformOLastDimElastix< TElastix >
::ReadFromFile( void )
{
  elxout << "ReadFromFile" << std::endl;
  /** Variables. */
  ReducedDimensionInputPointType centerOfRotationPoint;
  centerOfRotationPoint.Fill( 0.0 );
  bool pointRead = false;

  /** Try first to read the CenterOfRotationPoint from the
   * transform parameter file, this is the new, and preferred
   * way, since elastix 3.402.
   */
  pointRead = this->ReadCenterOfRotationPoint( centerOfRotationPoint );

  if( !pointRead )
  {
    xl::xout[ "error" ] << "ERROR: No center of rotation is specified in "
                        << "the transform parameter file" << std::endl;
    itkExceptionMacro( << "Transform parameter file is corrupt." );
  }

  /** Set the center in this Transform. */
  this->m_AffineLogTransformOLastDim->SetCenter( centerOfRotationPoint );

  /** Call the ReadFromFile from the TransformBase.
   * BE AWARE: Only call Superclass2::ReadFromFile() after CenterOfRotation
   * is set, because it is used in the SetParameters()-function of this transform.
   */
  this->Superclass2::ReadFromFile();

} // end ReadFromFile()


/**
 * ************************* WriteToFile ************************
 */

template< class TElastix >
void
AffineLogTransformOLastDimElastix< TElastix >
::WriteToFile( const ParametersType & param ) const
{
  elxout << "WriteToFile" << std::endl;
  /** Call the WriteToFile from the TransformBase. */
  this->Superclass2::WriteToFile( param );

  /** Write AffineLogTransformOLastDim specific things. */
  xout[ "transpar" ] << std::endl << "// AffineLogTransformOLastDim specific" << std::endl;

  /** Set the precision of cout to 10. */
  xout[ "transpar" ] << std::setprecision( 10 );

  /** Get the center of rotation point and write it to file. */
  ReducedDimensionInputPointType rotationPoint = this->m_AffineLogTransformOLastDim->GetCenter();
  xout[ "transpar" ] << "(CenterOfRotationPoint ";
  for( unsigned int i = 0; i < ReducedSpaceDimension - 1; i++ )
  {
    xout[ "transpar" ] << rotationPoint[ i ] << " ";
  }
  xout[ "transpar" ] << rotationPoint[ReducedSpaceDimension - 1 ] << ")" << std::endl;

  xout[ "transpar" ] << "(MatrixTranslation";
  for( unsigned int i = 0; i < ReducedSpaceDimension; ++i )
  {
    for( unsigned int j = 0; j < ReducedSpaceDimension; ++j )
    {
      xout[ "transpar" ] << " " << this->m_AffineLogTransformOLastDim->GetMatrix() ( i, j );
    }
  }
  for( unsigned int i = 0; i < ReducedSpaceDimension; ++i )
  {
    xout[ "transpar" ] << " " << this->m_AffineLogTransformOLastDim->GetTranslation()[ i ];
  }
  xout[ "transpar" ] << ")" << std::endl;

  /** Set the precision back to default value. */
  xout[ "transpar" ] << std::setprecision( this->m_Elastix->GetDefaultOutputPrecision() );

} // end WriteToFile()


/**
 * ************************* InitializeTransform *********************
 */

template< class TElastix >
void
AffineLogTransformOLastDimElastix< TElastix >
::InitializeTransform( void )
{
  elxout << "InitializeTransform" << std::endl;
  /** Set all parameters to zero (no rotations, no translation). */
  this->m_AffineLogTransformOLastDim->SetIdentity();

  /** Try to read CenterOfRotationIndex from parameter file,
   * which is the rotationPoint, expressed in index-values.
   */

  ContinuousIndexType                 centerOfRotationIndex;
  InputPointType                      centerOfRotationPoint;
  ReducedDimensionContinuousIndexType RDcenterOfRotationIndex;
  ReducedDimensionInputPointType      RDcenterOfRotationPoint;
  InputPointType                      TransformedCenterOfRotation;
  ReducedDimensionInputPointType      RDTransformedCenterOfRotation;

  bool     centerGivenAsIndex = true;
  bool     centerGivenAsPoint = true;
  SizeType fixedImageSize     = this->m_Registration->GetAsITKBaseType()->
    GetFixedImage()->GetLargestPossibleRegion().GetSize();

  for( unsigned int i = 0; i < ReducedSpaceDimension; i++ )
  {
    /** Initialize. */
    centerOfRotationIndex[ i ]         = 0;
    RDcenterOfRotationIndex[ i ]       = 0;
    RDcenterOfRotationPoint[ i ]       = 0.0;
    centerOfRotationPoint[ i ]         = 0.0;
    TransformedCenterOfRotation[ i ]   = 0.0;
    RDTransformedCenterOfRotation[ i ] = 0.0;

    /** Check COR index: Returns zero when parameter was in the parameter file. */
    bool foundI = this->m_Configuration->ReadParameter(
      RDcenterOfRotationIndex[ i ], "CenterOfRotation", i, false );
    if( !foundI )
    {
      centerGivenAsIndex &= false;
    }

    /** Check COR point: Returns zero when parameter was in the parameter file. */
    bool foundP = this->m_Configuration->ReadParameter(
      RDcenterOfRotationPoint[ i ], "CenterOfRotationPoint", i, false );
    if( !foundP )
    {
      centerGivenAsPoint &= false;
    }
    centerOfRotationPoint[i] = RDcenterOfRotationPoint[i];
    centerOfRotationIndex[i] = RDcenterOfRotationIndex[i];
  } // end loop over ReducedSpaceDimension

  // force 4D point for after
  centerOfRotationPoint[SpaceDimension - 1] = 0.0;
  centerOfRotationIndex[SpaceDimension - 1] = 0;
  TransformedCenterOfRotation[SpaceDimension - 1] = 0.0;

  /** Check if user wants automatic transform initialization; false by default.
   * If an initial transform is given, automatic transform initialization is
   * not possible.
   */
  bool automaticTransformInitialization = false;
  bool tmpBool                          = false;
  this->m_Configuration->ReadParameter( tmpBool,
    "AutomaticTransformInitialization", 0 );
  if( tmpBool && this->Superclass1::GetInitialTransform() == 0 )
  {
    automaticTransformInitialization = true;
  }

  /** Set the center of rotation to the center of the image if no center was given */
  bool centerGiven = centerGivenAsIndex || centerGivenAsPoint;
  if( !centerGiven  )
  {
    /** Use center of image as default center of rotation */
    for( unsigned int k = 0; k < ReducedSpaceDimension; k++ )
    {
      centerOfRotationIndex[ k ] = ( fixedImageSize[ k ] - 1.0 ) / 2.0;
    }
    /** Convert from continuous index to physical point */
    this->m_Registration->GetAsITKBaseType()->GetFixedImage()->
      TransformContinuousIndexToPhysicalPoint( centerOfRotationIndex, TransformedCenterOfRotation );

    for( unsigned int k = 0; k < ReducedSpaceDimension; k++ )
    {
      RDTransformedCenterOfRotation[ k ] = TransformedCenterOfRotation[ k ];
    }

    this->m_AffineLogTransformOLastDim->SetCenter( RDTransformedCenterOfRotation );
  }

  /** Set the translation to zero, if no AutomaticTransformInitialization
   * was desired.
   */
  if( !automaticTransformInitialization )
  {
    OutputVectorType noTranslation;
    noTranslation.Fill( 0.0 );
    this->m_AffineLogTransformOLastDim->SetTranslation( noTranslation );
  }
  if( centerGivenAsIndex )
  {
    this->m_Registration->GetAsITKBaseType()->GetFixedImage()
      ->TransformContinuousIndexToPhysicalPoint(
      centerOfRotationIndex, TransformedCenterOfRotation );
    for( unsigned int k = 0; k < ReducedSpaceDimension; k++ )
    {
      RDTransformedCenterOfRotation[ k ] = TransformedCenterOfRotation[ k ];
    }
    this->m_AffineLogTransformOLastDim->SetCenter( RDTransformedCenterOfRotation );
  }

  /** Set the translation to zero */
  ReducedDimensionOutputVectorType noTranslation;
  noTranslation.Fill( 0.0 );
  this->m_AffineLogTransformOLastDim->SetTranslation( noTranslation );


  /** Set the initial parameters in this->m_Registration. */
  this->m_Registration->GetAsITKBaseType()->
    SetInitialTransformParameters( this->GetParameters() );

} // end InitializeTransform()


/**
 * ************************* SetScales *********************
 */

template< class TElastix >
void
AffineLogTransformOLastDimElastix< TElastix >
::SetScales( void )
{
  elxout << "SetScales" << std::endl;
  /** Create the new scales. */
  const NumberOfParametersType N = this->GetNumberOfParameters();
  ScalesType                   newscales( N );
  newscales.Fill( 1.0 );

  /** Always estimate scales automatically */
  elxout << "Scales are estimated automatically." << std::endl;
  this->AutomaticScalesEstimationOLastDim( newscales );

  std::size_t count
    = this->m_Configuration->CountNumberOfParameterEntries( "Scales" );

  if( count == this->GetNumberOfParameters() )
  {
    /** Overrule the automatically estimated scales with the user-specified
     * scales. Values <= 0 are not used; the default is kept then. */
    for( unsigned int i = 0; i < this->GetNumberOfParameters(); i++ )
    {
      double scale_i = -1.0;
      this->m_Configuration->ReadParameter( scale_i, "Scales", i );
      if( scale_i > 0 )
      {
        newscales[ i ] = scale_i;
      }
    }
  }
  else if( count != 0 )
  {
    /** In this case an error is made in the parameter-file.
     * An error is thrown, because using erroneous scales in the optimizer
     * can give unpredictable results.
     */
    itkExceptionMacro( << "ERROR: The Scales-option in the parameter-file"
                       << " has not been set properly." );
  }

  elxout << "Scales for transform parameters are: " << newscales << std::endl;

  /** Set the scales into the optimizer. */
  this->m_Registration->GetAsITKBaseType()->GetModifiableOptimizer()->SetScales( newscales );

} // end SetScales()


/**
 * ******************** ReadCenterOfRotationPoint *********************
 */

template< class TElastix >
bool
AffineLogTransformOLastDimElastix< TElastix >
::ReadCenterOfRotationPoint( ReducedDimensionInputPointType & rotationPoint ) const
{
  elxout << "ReadCenterOfRotationPoint" << std::endl;
  /** Try to read CenterOfRotationPoint from the transform parameter
   * file, which is the rotationPoint, expressed in world coordinates.
   */
  ReducedDimensionInputPointType centerOfRotationPoint;
  bool           centerGivenAsPoint = true;
  for( unsigned int i = 0; i < ReducedSpaceDimension; i++ )
  {
    centerOfRotationPoint[ i ] = 0;

    /** Returns zero when parameter was in the parameter file */
    bool found = this->m_Configuration->ReadParameter(
      centerOfRotationPoint[ i ], "CenterOfRotationPoint", i, false );
    if( !found )
    {
      centerGivenAsPoint &= false;
    }
  }

  if( !centerGivenAsPoint )
  {
    return false;
  }

  /** Copy the temporary variable into the output of this function,
   * if everything went ok.
   */
  rotationPoint = centerOfRotationPoint;

  /** Successfully read centerOfRotation as Point. */
  return true;

} // end ReadCenterOfRotationPoint()



} // end namespace elastix

#endif // ELXAFFINELOGTRANSFORMOLASTDIM_HXX
