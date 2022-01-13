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
#ifndef __itkAffineLogTransformSongshu_h
#define __itkAffineLogTransformSongshu_h

#include <iostream>
#include "itkAdvancedMatrixOffsetTransformBase.h"

namespace itk
{

/** \class AffineLogTransformSongshu
 *
 *
 * \ingroup Transforms
 */
template< class TScalarType = double, unsigned int Dimension = 2 > // Data type for scalars (float or double)
class AffineLogTransformSongshu :
  public AdvancedMatrixOffsetTransformBase< TScalarType, Dimension, Dimension >
{
public:

  /** Standard class typedefs. */
  typedef AffineLogTransformSongshu                                              Self;
  typedef AdvancedMatrixOffsetTransformBase< TScalarType, Dimension, Dimension > Superclass;
  typedef AdvancedMatrixOffsetTransformBase< TScalarType, Dimension+1, Dimension+1 > UnReducedSuperclass;
  typedef SmartPointer< Self >                                                   Pointer;
  typedef SmartPointer< const Self >                                             ConstPointer;

  /** New macro for creation of through a Smart Pointer. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( AffineLogTransformSongshu, AdvancedMatrixOffsetTransformBase );

  /** Dimension of the domain space. */
  itkStaticConstMacro( SpaceDimension, unsigned int, Dimension );
  itkStaticConstMacro( OutputSpaceDimension, unsigned int, Dimension );
  itkStaticConstMacro( InputSpaceDimension, unsigned int, Dimension );
  itkStaticConstMacro( ParametersDimension, unsigned int, ( Dimension + 1 ) * Dimension );

  typedef typename Superclass::ParametersType            ParametersType;
  typedef typename Superclass::NumberOfParametersType    NumberOfParametersType;
  typedef typename Superclass::JacobianType              JacobianType;
  typedef typename Superclass::ScalarType                ScalarType;
  typedef typename Superclass::InputVectorType           InputVectorType;
  typedef typename Superclass::OutputVectorType          OutputVectorType;
  typedef typename Superclass::InputCovariantVectorType  InputCovariantVectorType;
  typedef typename Superclass::OutputCovariantVectorType OutputCovariantVectorType;
  typedef typename Superclass::InputVnlVectorType        InputVnlVectorType;
  typedef typename Superclass::OutputVnlVectorType       OutputVnlVectorType;
  typedef typename Superclass::InputPointType            InputPointType;
  typedef typename UnReducedSuperclass::InputPointType   UnReducedInputPointType;
  typedef typename Superclass::OutputPointType           OutputPointType;
  typedef typename Superclass::MatrixType                MatrixType;
  typedef typename Superclass::InverseMatrixType         InverseMatrixType;
  typedef typename Superclass::CenterType                CenterType;
  typedef typename Superclass::TranslationType           TranslationType;
  typedef typename Superclass::OffsetType                OffsetType;
  typedef typename Superclass::ScalarType                AngleType;

  typedef typename Superclass
    ::NonZeroJacobianIndicesType NonZeroJacobianIndicesType;
  typedef typename Superclass::SpatialJacobianType SpatialJacobianType;
  typedef typename Superclass
    ::JacobianOfSpatialJacobianType JacobianOfSpatialJacobianType;
  typedef typename Superclass::SpatialHessianType SpatialHessianType;
  typedef typename Superclass
    ::JacobianOfSpatialHessianType JacobianOfSpatialHessianType;
  typedef typename Superclass::InternalMatrixType InternalMatrixType;

  typedef FixedArray< ScalarType > ScalarArrayType;

  void SetParameters( const ParametersType & parameters ) override;

  const ParametersType & GetParameters( void ) const override;

  /** Compute the Jacobian of the transformation. */
  void GetJacobian(
    const InputPointType &,
    JacobianType &,
    NonZeroJacobianIndicesType & ) const override;

  void SetIdentity( void ) override;

protected:

  AffineLogTransformSongshu();
  AffineLogTransformSongshu( const MatrixType & matrix,
    const OutputPointType & offset );
  AffineLogTransformSongshu( unsigned int outputSpaceDims,
    unsigned int paramsSpaceDims );

  ~AffineLogTransformSongshu() override{}

  void PrintSelf( std::ostream & os, Indent indent ) const override;

  /** Update the m_JacobianOfSpatialJacobian.  */
  virtual void PrecomputeJacobianOfSpatialJacobian( void );

private:

  AffineLogTransformSongshu( const Self & ); // purposely not implemented
  void operator=( const Self & );     // purposely not implemented

  MatrixType m_MatrixLogDomain;

};

}  // namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkAffineLogTransformSongshu.hxx"
#endif

#endif /* __itkAffineLogTransformSongshu_h */