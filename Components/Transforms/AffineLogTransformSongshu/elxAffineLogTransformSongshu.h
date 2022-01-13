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
#ifndef _ELXAFFINELOGTRANSFORMSONGSHU_H_
#define _ELXAFFINELOGTRANSFORMSONGSHU_H_

#include "itkAdvancedCombinationTransform.h"
#include "itkAffineLogTransformSongshu.h"
#include "itkCenteredTransformInitializerSongshu.h"
#include "elxIncludes.h"
#include "itkMacro.h"

namespace elastix
{

/**
 * \class AffineLogTransformSongshuElastix
 * \brief
 *
 * This transform is an affine transformation, with a different parametrisation
 * than the usual one.
 *
 * \warning: the behaviour of this transform might still change in the future. It is still experimental.
 *
 * \ingroup Transforms
 * \sa AffineLogTransformSongshu
 */

template< class TElastix >
class AffineLogTransformSongshuElastix :
  public itk::AdvancedCombinationTransform<
    typename elx::TransformBase< TElastix >::CoordRepType,
    elx::TransformBase< TElastix >::ReducedImageDimension >,
  public elx::TransformBase< TElastix >
{
public:

  /** Standard ITK-stuff.*/
  typedef AffineLogTransformSongshuElastix                       Self;
  typedef itk::AdvancedCombinationTransform<
    typename elx::TransformBase< TElastix >::CoordRepType,
    elx::TransformBase< TElastix >:: ReducedImageDimension > Superclass1;
  typedef elx::TransformBase< TElastix >                  Superclass2;
  typedef itk::SmartPointer< Self >                       Pointer;
  typedef itk::SmartPointer< const Self >                 ConstPointer;

  /** The ITK-class that provides most of the functionality, and
   * that is set as the "CurrentTransform" in the CombinationTransform */
  typedef itk::AffineLogTransformSongshu<
    typename elx::TransformBase< TElastix >::CoordRepType,
    elx::TransformBase< TElastix >::ReducedImageDimension >     AffineLogTransformType;

  /** Method for creation through the object factory. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( AffineLogTransformSongshuElastix, AdvancedCombinationTransform );

  /** Name of this class.
   * Use this name in the parameter file to select this specific transform. \n
   * example: <tt>(Transform "AffineLogTransform")</tt>\n
   */
  elxClassNameMacro( "AffineLogTransformSongshu" );

  /** Dimension of the fixed image. */
  itkStaticConstMacro( SpaceDimension, unsigned int, Superclass2::FixedImageDimension );
  itkStaticConstMacro(ReducedSpaceDimension, unsigned int, Superclass2::FixedImageDimension - 1 );

  /** Typedefs inherited from the superclass. */
  typedef typename Superclass1::ScalarType             ScalarType;
  typedef typename Superclass1::ParametersType         ParametersType;
  typedef typename Superclass1::NumberOfParametersType NumberOfParametersType;
  typedef typename Superclass1::JacobianType           JacobianType;

  typedef typename AffineLogTransformType::InputPointType   ReducedDimensionInputPointType;
  typedef typename AffineLogTransformType::OutputVectorType ReducedDimensionOutputVectorType;
  typedef typename AffineLogTransformType::UnReducedInputPointType   InputPointType;
 
  typedef typename Superclass1::OutputPointType           OutputPointType;
  typedef typename Superclass1::InputVectorType           InputVectorType;
  typedef typename Superclass1::OutputVectorType          OutputVectorType;
  typedef typename Superclass1::InputCovariantVectorType  InputCovariantVectorType;
  typedef typename Superclass1::OutputCovariantVectorType OutputCovariantVectorType;
  typedef typename Superclass1::InputVnlVectorType        InputVnlVectorType;
  typedef typename Superclass1::OutputVnlVectorType       OutputVnlVectorType;

  typedef typename AffineLogTransformType::Pointer    AffineLogTransformPointer;
  typedef typename AffineLogTransformType::OffsetType OffsetType;

  /** Typedef's inherited from TransformBase. */
  typedef typename Superclass2::ElastixType              ElastixType;
  typedef typename Superclass2::ElastixPointer           ElastixPointer;
  typedef typename Superclass2::ConfigurationType        ConfigurationType;
  typedef typename Superclass2::ConfigurationPointer     ConfigurationPointer;
  typedef typename Superclass2::RegistrationType         RegistrationType;
  typedef typename Superclass2::RegistrationPointer      RegistrationPointer;
  typedef typename Superclass2::CoordRepType             CoordRepType;
  typedef typename Superclass2::FixedImageType           FixedImageType;
  typedef typename Superclass2::MovingImageType          MovingImageType;
  typedef typename Superclass2::ITKBaseType              ITKBaseType;
  typedef typename Superclass2::CombinationTransformType CombinationTransformType;

  /** Reduced Dimension typedef's. */
  typedef float PixelType;
  typedef itk::ImageRegion<
    itkGetStaticConstMacro(ReducedSpaceDimension) >         ReducedDimensionRegionType;
  typedef itk::Image< PixelType,
    itkGetStaticConstMacro( ReducedSpaceDimension ) >       ReducedDimensionImageType;
  typedef typename ReducedDimensionRegionType::IndexType    ReducedDimensionIndexType;
  typedef typename ReducedDimensionImageType::SizeType      ReducedDimensionSizeType;
  typedef typename ReducedDimensionImageType::PointType     ReducedDimensionPointType;
  typedef typename ReducedDimensionImageType::SpacingType   ReducedDimensionSpacingType;

  typedef typename ReducedDimensionImageType::DirectionType ReducedDimensionDirectionType;

  typedef typename ReducedDimensionImageType::PointType     ReducedDimensionOriginType;


  /** Other typedef's. */
  typedef typename FixedImageType::IndexType     IndexType;
  typedef typename IndexType::IndexValueType     IndexValueType;
  typedef typename FixedImageType::SizeType      SizeType;
  typedef typename FixedImageType::PointType     PointType;
  typedef typename FixedImageType::SpacingType   SpacingType;
  typedef typename FixedImageType::RegionType    RegionType;
  typedef typename FixedImageType::DirectionType DirectionType;
  typedef typename itk::ContinuousIndex< CoordRepType, ReducedSpaceDimension > ReducedDimensionContinuousIndexType;
  typedef typename itk::ContinuousIndex< CoordRepType, SpaceDimension >        ContinuousIndexType;

  typedef itk::CenteredTransformInitializerSongshu<
    AffineLogTransformType, FixedImageType, MovingImageType >  TransformInitializerType;
  typedef typename TransformInitializerType::Pointer TransformInitializerPointer;

  /** For scales setting in the optimizer */
  typedef typename Superclass2::ScalesType ScalesType;

  /** Execute stuff before the actual registration:
   * \li Call InitializeTransform
   * \li Set the scales.
   */
  void BeforeRegistration( void ) override;

  /** Initialize Transform.
   * \li Set all parameters to zero.
   * \li Set center of rotation:
   *  automatically initialized to the geometric center of the image, or
   *   assigned a user entered voxel index, given by the parameter
   *   (CenterOfRotation <index-x> <index-y> ...);
   *   If an initial transform is present and HowToCombineTransforms is
   *   set to "Compose", the initial transform is taken into account
   *   while setting the center of rotation.
   * \li Set initial translation:
   *  the initial translation between fixed and moving image is guessed,
   *  if the user has set (AutomaticTransformInitialization "true").
   *
   * It is not yet possible to enter an initial rotation angle.
   */
  virtual void InitializeTransform( void );

  /** Set the scales
   * \li If AutomaticScalesEstimation is "true" estimate scales
   * \li If scales are provided by the user use those,
   * \li Otherwise use some default value
   * This function is called by BeforeRegistration, after
   * the InitializeTransform function is called
   */
  virtual void SetScales( void );

  /** Function to read transform-parameters from a file.
   *
   * It reads the center of rotation and calls the superclass' implementation.
   */
  void ReadFromFile( void ) override;

  /** Function to write transform-parameters to a file.
   * It writes the center of rotation to file and calls the superclass' implementation.
   */
  void WriteToFile( const ParametersType & param ) const override;

protected:

  /** The constructor. */
  AffineLogTransformSongshuElastix();

  /** The destructor. */
  ~AffineLogTransformSongshuElastix() override{}

  /** Try to read the CenterOfRotationPoint from the transform parameter file
   * The CenterOfRotationPoint is already in world coordinates. */
  virtual bool ReadCenterOfRotationPoint( ReducedDimensionInputPointType & rotationPoint ) const;

private:

  /** The private constructor. */
  AffineLogTransformSongshuElastix( const Self & );  // purposely not implemented
  /** The private copy constructor. */
  void operator=( const Self & );         // purposely not implemented

  AffineLogTransformPointer m_AffineLogTransform;

};

} // end namespace elastix

#endif // ELXAFFINELOGTRANSFORMSONGSHU_H

#ifndef ITK_MANUAL_INSTANTIATION
#include "elxAffineLogTransformSongshu.hxx"
#endif