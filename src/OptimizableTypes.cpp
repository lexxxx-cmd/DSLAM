/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/

#include "OptimizableTypes.h"

namespace ORB_SLAM3 {
    Eigen::Vector3d EdgeStereoSE3ProjectXYZOnlyPose::cam_project(const Eigen::Vector3d & trans_xyz) const{
        const float invz = 1.0f/trans_xyz[2];
        Eigen::Vector3d res;
        res[0] = trans_xyz[0]*invz*fx + cx;
        res[1] = trans_xyz[1]*invz*fy + cy;
        res[2] = res[0] - bf*invz;
        return res;
    }


    bool EdgeStereoSE3ProjectXYZOnlyPose::read(std::istream& is){
        for (int i=0; i<=3; i++){
            is >> _measurement[i];
        }
        for (int i=0; i<=2; i++)
            for (int j=i; j<=2; j++) {
            is >> information()(i,j);
            if (i!=j)
                information()(j,i)=information()(i,j);
            }
        return true;
    }

    bool EdgeStereoSE3ProjectXYZOnlyPose::write(std::ostream& os) const {

        for (int i=0; i<=3; i++){
            os << measurement()[i] << " ";
        }

        for (int i=0; i<=2; i++)
            for (int j=i; j<=2; j++){
            os << " " <<  information()(i,j);
            }
        return os.good();
    }

    void EdgeStereoSE3ProjectXYZOnlyPose::linearizeOplus() {
        g2o::VertexSE3Expmap * vi = static_cast<g2o::VertexSE3Expmap *>(_vertices[0]);
        Eigen::Vector3d xyz_trans = vi->estimate().map(Xw);

        double x = xyz_trans[0];
        double y = xyz_trans[1];
        double invz = 1.0/xyz_trans[2];
        double invz_2 = invz*invz;

        _jacobianOplusXi(0,0) =  x*y*invz_2 *fx;
        _jacobianOplusXi(0,1) = -(1+(x*x*invz_2)) *fx;
        _jacobianOplusXi(0,2) = y*invz *fx;
        _jacobianOplusXi(0,3) = -invz *fx;
        _jacobianOplusXi(0,4) = 0;
        _jacobianOplusXi(0,5) = x*invz_2 *fx;

        _jacobianOplusXi(1,0) = (1+y*y*invz_2) *fy;
        _jacobianOplusXi(1,1) = -x*y*invz_2 *fy;
        _jacobianOplusXi(1,2) = -x*invz *fy;
        _jacobianOplusXi(1,3) = 0;
        _jacobianOplusXi(1,4) = -invz *fy;
        _jacobianOplusXi(1,5) = y*invz_2 *fy;

        _jacobianOplusXi(2,0) = _jacobianOplusXi(0,0)-bf*y*invz_2;
        _jacobianOplusXi(2,1) = _jacobianOplusXi(0,1)+bf*x*invz_2;
        _jacobianOplusXi(2,2) = _jacobianOplusXi(0,2);
        _jacobianOplusXi(2,3) = _jacobianOplusXi(0,3);
        _jacobianOplusXi(2,4) = 0;
        _jacobianOplusXi(2,5) = _jacobianOplusXi(0,5)-bf*invz_2;
        _jacobianOplusXi = weight * _jacobianOplusXi;
    }
    Eigen::Vector3d EdgeStereoSE3ProjectXYZ::cam_project(const Eigen::Vector3d & trans_xyz, const float &bf) const{
        const float invz = 1.0f/trans_xyz[2];
        Eigen::Vector3d res;
        res[0] = trans_xyz[0]*invz*fx + cx;
        res[1] = trans_xyz[1]*invz*fy + cy;
        res[2] = res[0] - bf*invz;
        return res;
    }

    EdgeStereoSE3ProjectXYZ::EdgeStereoSE3ProjectXYZ(float w) : g2o::BaseBinaryEdge<3, Eigen::Vector3d, g2o::VertexSBAPointXYZ, g2o::VertexSE3Expmap>() {
        weight = w;
    }

    bool EdgeStereoSE3ProjectXYZ::read(std::istream& is){
        for (int i=0; i<=3; i++){
            is >> _measurement[i];
        }
        for (int i=0; i<=2; i++)
            for (int j=i; j<=2; j++) {
            is >> information()(i,j);
            if (i!=j)
                information()(j,i)=information()(i,j);
            }
        return true;
    }

    bool EdgeStereoSE3ProjectXYZ::write(std::ostream& os) const {

        for (int i=0; i<=3; i++){
            os << measurement()[i] << " ";
        }

        for (int i=0; i<=2; i++)
            for (int j=i; j<=2; j++){
            os << " " <<  information()(i,j);
            }
        return os.good();
    }

    void EdgeStereoSE3ProjectXYZ::linearizeOplus() {
        g2o::VertexSE3Expmap * vj = static_cast<g2o::VertexSE3Expmap *>(_vertices[1]);
        g2o::SE3Quat T(vj->estimate());
        g2o::VertexSBAPointXYZ* vi = static_cast<g2o::VertexSBAPointXYZ*>(_vertices[0]);
        Eigen::Vector3d xyz = vi->estimate();
        Eigen::Vector3d xyz_trans = T.map(xyz);

        const Eigen::Matrix3d R =  T.rotation().toRotationMatrix();

        double x = xyz_trans[0];
        double y = xyz_trans[1];
        double z = xyz_trans[2];
        double z_2 = z*z;

        _jacobianOplusXi(0,0) = -fx*R(0,0)/z+fx*x*R(2,0)/z_2;
        _jacobianOplusXi(0,1) = -fx*R(0,1)/z+fx*x*R(2,1)/z_2;
        _jacobianOplusXi(0,2) = -fx*R(0,2)/z+fx*x*R(2,2)/z_2;

        _jacobianOplusXi(1,0) = -fy*R(1,0)/z+fy*y*R(2,0)/z_2;
        _jacobianOplusXi(1,1) = -fy*R(1,1)/z+fy*y*R(2,1)/z_2;
        _jacobianOplusXi(1,2) = -fy*R(1,2)/z+fy*y*R(2,2)/z_2;

        _jacobianOplusXi(2,0) = _jacobianOplusXi(0,0)-bf*R(2,0)/z_2;
        _jacobianOplusXi(2,1) = _jacobianOplusXi(0,1)-bf*R(2,1)/z_2;
        _jacobianOplusXi(2,2) = _jacobianOplusXi(0,2)-bf*R(2,2)/z_2;

        _jacobianOplusXj(0,0) =  x*y/z_2 *fx;
        _jacobianOplusXj(0,1) = -(1+(x*x/z_2)) *fx;
        _jacobianOplusXj(0,2) = y/z *fx;
        _jacobianOplusXj(0,3) = -1./z *fx;
        _jacobianOplusXj(0,4) = 0;
        _jacobianOplusXj(0,5) = x/z_2 *fx;

        _jacobianOplusXj(1,0) = (1+y*y/z_2) *fy;
        _jacobianOplusXj(1,1) = -x*y/z_2 *fy;
        _jacobianOplusXj(1,2) = -x/z *fy;
        _jacobianOplusXj(1,3) = 0;
        _jacobianOplusXj(1,4) = -1./z *fy;
        _jacobianOplusXj(1,5) = y/z_2 *fy;

        _jacobianOplusXj(2,0) = _jacobianOplusXj(0,0)-bf*y/z_2;
        _jacobianOplusXj(2,1) = _jacobianOplusXj(0,1)+bf*x/z_2;
        _jacobianOplusXj(2,2) = _jacobianOplusXj(0,2);
        _jacobianOplusXj(2,3) = _jacobianOplusXj(0,3);
        _jacobianOplusXj(2,4) = 0;
        _jacobianOplusXj(2,5) = _jacobianOplusXj(0,5)-bf/z_2;
        _jacobianOplusXi *= weight;
        _jacobianOplusXj *= weight;
    }

    bool EdgeSE3ProjectXYZOnlyPose::read(std::istream& is){
        for (int i=0; i<2; i++){
            is >> _measurement[i];
        }
        for (int i=0; i<2; i++)
            for (int j=i; j<2; j++) {
                is >> information()(i,j);
                if (i!=j)
                    information()(j,i)=information()(i,j);
            }
        return true;
    }

    bool EdgeSE3ProjectXYZOnlyPose::write(std::ostream& os) const {

        for (int i=0; i<2; i++){
            os << measurement()[i] << " ";
        }

        for (int i=0; i<2; i++)
            for (int j=i; j<2; j++){
                os << " " <<  information()(i,j);
            }
        return os.good();
    }


    void EdgeSE3ProjectXYZOnlyPose::linearizeOplus() {
        g2o::VertexSE3Expmap * vi = static_cast<g2o::VertexSE3Expmap *>(_vertices[0]);
        Eigen::Vector3d xyz_trans = vi->estimate().map(Xw);

        double x = xyz_trans[0];
        double y = xyz_trans[1];
        double z = xyz_trans[2];

        Eigen::Matrix<double,3,6> SE3deriv;
        SE3deriv << 0.f, z,   -y, 1.f, 0.f, 0.f,
                     -z , 0.f, x, 0.f, 1.f, 0.f,
                     y ,  -x , 0.f, 0.f, 0.f, 1.f;

        _jacobianOplusXi = -pCamera->projectJac(xyz_trans) * SE3deriv;
    }
    void EdgeSE3Derect::linearizeOplus() {
        if(level() == 1) {
            _jacobianOplusXi = Eigen::Matrix<double, 1, 6>::Zero();
            return;
        }

        g2o::VertexSE3Expmap* vtx = dynamic_cast<g2o::VertexSE3Expmap*> ( _vertices[0] );
        Eigen::Vector3d xyz_trans = vtx->estimate().map ( Xw );
        Eigen::Vector2d uv = pCamera->project(xyz_trans) * pCamera->getScale();
        float u = uv.x(), v = uv.y();


        //计算雅可比
        Eigen::Matrix<double, 1, 2> jacobian_I_uv;
        jacobian_I_uv (0,0) = ( getPixelValue ( u+1,v )-getPixelValue ( u-1,v ) ) /2;
        jacobian_I_uv ( 0,1 ) = ( getPixelValue ( u,v+1 )-getPixelValue ( u,v-1 ) ) /2;

        double x = xyz_trans[0];
        double y = xyz_trans[1];
        double invz = 1.0/xyz_trans[2];
        double invz_2 = invz * invz;
        float fx = pCamera->getParameter(0) * pCamera->getScale();
        float fy = pCamera->getParameter(1) * pCamera->getScale();
        Eigen::Matrix<double, 2, 6> jacobian_uv_ksai;
        jacobian_uv_ksai ( 0,0 ) = invz *fx;
        jacobian_uv_ksai ( 0,1 ) = 0;
        jacobian_uv_ksai ( 0,2 ) = -x*invz_2 *fx;
        jacobian_uv_ksai ( 0,3 ) = - x*y*invz_2 *fx;
        jacobian_uv_ksai ( 0,4 ) = ( 1+ ( x*x*invz_2 ) ) *fx;
        jacobian_uv_ksai ( 0,5 ) = - y*invz *fx;

        jacobian_uv_ksai ( 1,0 ) = 0;
        jacobian_uv_ksai ( 1,1 ) = invz *fy;
        jacobian_uv_ksai ( 1,2 ) = -y*invz_2 *fy;
        jacobian_uv_ksai ( 1,3 ) = - ( 1+y*y*invz_2 ) *fy;
        jacobian_uv_ksai ( 1,4 ) = x*y*invz_2 *fy;
        jacobian_uv_ksai ( 1,5 ) = x*invz *fy;

        _jacobianOplusXi = jacobian_I_uv*jacobian_uv_ksai;
    }
    bool EdgeSE3ProjectXYZOnlyPoseToBody::read(std::istream& is){
        for (int i=0; i<2; i++){
            is >> _measurement[i];
        }
        for (int i=0; i<2; i++)
            for (int j=i; j<2; j++) {
                is >> information()(i,j);
                if (i!=j)
                    information()(j,i)=information()(i,j);
            }
        return true;
    }

    bool EdgeSE3ProjectXYZOnlyPoseToBody::write(std::ostream& os) const {

        for (int i=0; i<2; i++){
            os << measurement()[i] << " ";
        }

        for (int i=0; i<2; i++)
            for (int j=i; j<2; j++){
                os << " " <<  information()(i,j);
            }
        return os.good();
    }

    void EdgeSE3ProjectXYZOnlyPoseToBody::linearizeOplus() {
        g2o::VertexSE3Expmap * vi = static_cast<g2o::VertexSE3Expmap *>(_vertices[0]);
        g2o::SE3Quat T_lw(vi->estimate());
        Eigen::Vector3d X_l = T_lw.map(Xw);
        Eigen::Vector3d X_r = mTrl.map(T_lw.map(Xw));

        double x_w = X_l[0];
        double y_w = X_l[1];
        double z_w = X_l[2];

        Eigen::Matrix<double,3,6> SE3deriv;
        SE3deriv << 0.f, z_w,   -y_w, 1.f, 0.f, 0.f,
                -z_w , 0.f, x_w, 0.f, 1.f, 0.f,
                y_w ,  -x_w , 0.f, 0.f, 0.f, 1.f;

        _jacobianOplusXi = -pCamera->projectJac(X_r) * mTrl.rotation().toRotationMatrix() * SE3deriv;
    }

    EdgeSE3ProjectXYZ::EdgeSE3ProjectXYZ() : BaseBinaryEdge<2, Eigen::Vector2d, g2o::VertexSBAPointXYZ, g2o::VertexSE3Expmap>() {
    }

    bool EdgeSE3ProjectXYZ::read(std::istream& is){
        for (int i=0; i<2; i++){
            is >> _measurement[i];
        }
        for (int i=0; i<2; i++)
            for (int j=i; j<2; j++) {
                is >> information()(i,j);
                if (i!=j)
                    information()(j,i)=information()(i,j);
            }
        return true;
    }

    bool EdgeSE3ProjectXYZ::write(std::ostream& os) const {

        for (int i=0; i<2; i++){
            os << measurement()[i] << " ";
        }

        for (int i=0; i<2; i++)
            for (int j=i; j<2; j++){
                os << " " <<  information()(i,j);
            }
        return os.good();
    }


    void EdgeSE3ProjectXYZ::linearizeOplus() {
        g2o::VertexSE3Expmap * vj = static_cast<g2o::VertexSE3Expmap *>(_vertices[1]);
        g2o::SE3Quat T(vj->estimate());
        g2o::VertexSBAPointXYZ* vi = static_cast<g2o::VertexSBAPointXYZ*>(_vertices[0]);
        Eigen::Vector3d xyz = vi->estimate();
        Eigen::Vector3d xyz_trans = T.map(xyz);

        double x = xyz_trans[0];
        double y = xyz_trans[1];
        double z = xyz_trans[2];

        auto projectJac = -pCamera->projectJac(xyz_trans);

        _jacobianOplusXi =  projectJac * T.rotation().toRotationMatrix();

        Eigen::Matrix<double,3,6> SE3deriv;
        SE3deriv << 0.f, z,   -y, 1.f, 0.f, 0.f,
                -z , 0.f, x, 0.f, 1.f, 0.f,
                y ,  -x , 0.f, 0.f, 0.f, 1.f;

        _jacobianOplusXj = projectJac * SE3deriv;
    }

    EdgeSE3ProjectXYZToBody::EdgeSE3ProjectXYZToBody() : BaseBinaryEdge<2, Eigen::Vector2d, g2o::VertexSBAPointXYZ, g2o::VertexSE3Expmap>() {
    }

    bool EdgeSE3ProjectXYZToBody::read(std::istream& is){
        for (int i=0; i<2; i++){
            is >> _measurement[i];
        }
        for (int i=0; i<2; i++)
            for (int j=i; j<2; j++) {
                is >> information()(i,j);
                if (i!=j)
                    information()(j,i)=information()(i,j);
            }
        return true;
    }

    bool EdgeSE3ProjectXYZToBody::write(std::ostream& os) const {

        for (int i=0; i<2; i++){
            os << measurement()[i] << " ";
        }

        for (int i=0; i<2; i++)
            for (int j=i; j<2; j++){
                os << " " <<  information()(i,j);
            }
        return os.good();
    }


    void EdgeSE3ProjectXYZToBody::linearizeOplus() {
        g2o::VertexSE3Expmap * vj = static_cast<g2o::VertexSE3Expmap *>(_vertices[1]);
        g2o::SE3Quat T_lw(vj->estimate());
        g2o::SE3Quat T_rw = mTrl * T_lw;
        g2o::VertexSBAPointXYZ* vi = static_cast<g2o::VertexSBAPointXYZ*>(_vertices[0]);
        Eigen::Vector3d X_w = vi->estimate();
        Eigen::Vector3d X_l = T_lw.map(X_w);
        Eigen::Vector3d X_r = mTrl.map(T_lw.map(X_w));

        _jacobianOplusXi =  -pCamera->projectJac(X_r) * T_rw.rotation().toRotationMatrix();

        double x = X_l[0];
        double y = X_l[1];
        double z = X_l[2];

        Eigen::Matrix<double,3,6> SE3deriv;
        SE3deriv << 0.f, z,   -y, 1.f, 0.f, 0.f,
                -z , 0.f, x, 0.f, 1.f, 0.f,
                y ,  -x , 0.f, 0.f, 0.f, 1.f;

        _jacobianOplusXj = -pCamera->projectJac(X_r) * mTrl.rotation().toRotationMatrix() * SE3deriv;
    }


    VertexSim3Expmap::VertexSim3Expmap() : BaseVertex<7, g2o::Sim3>()
    {
        _marginalized=false;
        _fix_scale = false;
    }

    bool VertexSim3Expmap::read(std::istream& is)
    {
        g2o::Vector7d cam2world;
        for (int i=0; i<6; i++){
            is >> cam2world[i];
        }
        is >> cam2world[6];

        float nextParam;
        for(size_t i = 0; i < pCamera1->size(); i++){
            is >> nextParam;
            pCamera1->setParameter(nextParam,i);
        }

        for(size_t i = 0; i < pCamera2->size(); i++){
            is >> nextParam;
            pCamera2->setParameter(nextParam,i);
        }

        setEstimate(g2o::Sim3(cam2world).inverse());
        return true;
    }

    bool VertexSim3Expmap::write(std::ostream& os) const
    {
        g2o::Sim3 cam2world(estimate().inverse());
        g2o::Vector7d lv=cam2world.log();
        for (int i=0; i<7; i++){
            os << lv[i] << " ";
        }

        for(size_t i = 0; i < pCamera1->size(); i++){
            os << pCamera1->getParameter(i) << " ";
        }

        for(size_t i = 0; i < pCamera2->size(); i++){
            os << pCamera2->getParameter(i) << " ";
        }

        return os.good();
    }

    EdgeSim3ProjectXYZ::EdgeSim3ProjectXYZ() :
            g2o::BaseBinaryEdge<2, Eigen::Vector2d, g2o::VertexSBAPointXYZ, VertexSim3Expmap>()
    {
    }

    bool EdgeSim3ProjectXYZ::read(std::istream& is)
    {
        for (int i=0; i<2; i++)
        {
            is >> _measurement[i];
        }

        for (int i=0; i<2; i++)
            for (int j=i; j<2; j++) {
                is >> information()(i,j);
                if (i!=j)
                    information()(j,i)=information()(i,j);
            }
        return true;
    }

    bool EdgeSim3ProjectXYZ::write(std::ostream& os) const
    {
        for (int i=0; i<2; i++){
            os  << _measurement[i] << " ";
        }

        for (int i=0; i<2; i++)
            for (int j=i; j<2; j++){
                os << " " <<  information()(i,j);
            }
        return os.good();
    }

    EdgeInverseSim3ProjectXYZ::EdgeInverseSim3ProjectXYZ() :
            g2o::BaseBinaryEdge<2, Eigen::Vector2d, g2o::VertexSBAPointXYZ, VertexSim3Expmap>()
    {
    }

    bool EdgeInverseSim3ProjectXYZ::read(std::istream& is)
    {
        for (int i=0; i<2; i++)
        {
            is >> _measurement[i];
        }

        for (int i=0; i<2; i++)
            for (int j=i; j<2; j++) {
                is >> information()(i,j);
                if (i!=j)
                    information()(j,i)=information()(i,j);
            }
        return true;
    }

    bool EdgeInverseSim3ProjectXYZ::write(std::ostream& os) const
    {
        for (int i=0; i<2; i++){
            os  << _measurement[i] << " ";
        }

        for (int i=0; i<2; i++)
            for (int j=i; j<2; j++){
                os << " " <<  information()(i,j);
            }
        return os.good();
    }

}
