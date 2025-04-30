// Copyright (c) 2020-2024, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto. Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.
//

#pragma once

#include "isaacsim/core/utils/Math.h"

#include <omni/physics/tensors/BodyTypes.h>
#include <omni/physx/IPhysxSceneQuery.h>

#include <DynamicControl.h>

namespace omni
{
namespace isaac
{
namespace surface_gripper
{


using namespace omni::isaac::dynamic_control;
using namespace isaacsim::core::utils::math;
using omni::physics::tensors::Transform;

inline const pxr::SdfPath& intToPath(const uint64_t& path)
{
    static_assert(sizeof(pxr::SdfPath) == sizeof(uint64_t), "Change to make the same size as pxr::SdfPath");

    return reinterpret_cast<const pxr::SdfPath&>(path);
}


/**
 * @brief Properties for Surface Gripper (suction-style gripper)
 *
 */
struct SurfaceGripperProperties
{
    std::string d6JointPath; //! USD path of the joint
    std::string parentPath; //! parent body that  contains the joint
    Transform offset; //! offset from parent body to joint point of contact in vacuum pressure
    float gripThreshold; //!  How far from an object it allows the gripper to lock in. Object will be pulled in this
                         //!  distance when gripper is closed
    float forceLimit; //! gripper breaking force
    float torqueLimit; //! torque breaking force
    float bendAngle; //! maximum bend angle for the gripper
    float stiffness; //! Gripper Stiffness
    float damping; //! Gripper damping
    bool disableGravity; //! flag to disable gravity of selected item to compensate for object's mass on robotic
                         //! controllers
    bool retryClose; //! Flag to indicate if gripper should keep attempting to close until it grips some object
};

/**
 * @brief Surface Gripper (suction-cup style gripper)
 *
 */
class SurfaceGripper
{

public:
    /**
     * @brief Creates a Surface Gripper
     *
     * @param[in] dc.
     * @param[in] props.
     */
    SurfaceGripper()
    {
        mDc = carb::getCachedInterface<omni::isaac::dynamic_control::DynamicControl>();
        mPhysxQuery = carb::getCachedInterface<omni::physx::IPhysxSceneQuery>();
        mJointProperties.body0 = 0;
        mJointProperties.axes = kDcAxisNone;
        mJointProperties.jointType = DcJointType::eSpherical;
        mJointProperties.stiffness = 0;
        mJointProperties.damping = 1.0e5f;
        mJointProperties.forceLimit = 0;
        mJointProperties.torqueLimit = 0;
        for (int i = 0; i < 6; ++i)
        {
            mJointProperties.hasLimits[i] = false;
        }

        mIsClosed = false;
        mIsInitialized = false;

        mJointHandle = omni::isaac::dynamic_control::kDcInvalidHandle;
    }

    /**
     * @brief Destroy the Surface Gripper object
     *
     */
    ~SurfaceGripper()
    {
        // Make sure that DC is valid before we destroy in case Dc was released already.
        if (mJointHandle && mDc)
        {
            mDc->destroyD6Joint(mJointHandle);
        }
    }

    /**
     * @brief Initialize joint and register with DC extension
     *
     * @param props
     * @return true
     * @return false
     */
    bool initialize(const SurfaceGripperProperties& props)
    {
        mProps = props;

        if (mJointHandle)
        {
            mDc->destroyD6Joint(mJointHandle);
            mJointHandle = omni::isaac::dynamic_control::kDcInvalidHandle;
        }
        mIsInitialized = true;
        return true;
    }

    /**
     * @brief returns whether the joint is closed
     * @return bool.
     */
    inline bool isClosed()
    {
        return mIsClosed;
    }

    /**
     * @brief returns whether the joint is closed
     * @return bool.
     */
    inline bool isAttemptingClose()
    {
        return mAttemptClose;
    }

    void update()
    {
        if (isClosed())
        {
            if (mDc->getD6JointConstraintIsBroken(mJointHandle))
            {
                //CARB_LOG_WARN("Gripper Constraint is Broken");
                open();
            }
        }
        else if (mAttemptClose)
        {
            attemptClose();
        }
    }

    /**
     * @brief closes the Surface Gripper, if any object is closer than the lock threshold,
     * otherwise it will keep attempting to close until it either picks some object, or open() is called
     *
     * @return true when an object is close and joint is effectively closed
     * @return false when no object is near the gripper. joint is not closed.
     */
    bool close()
    {
        if (!mIsInitialized)
        {
            CARB_LOG_ERROR("Please call initialize before closing");
            return false;
        }
        else
        {
            mAttemptClose = true;
            return attemptClose();
        }
        return false;
    }

    /**
     * @brief attempts closing the Surface Gripper, if any object is closer than the lock threshold
     *
     * @return true when an object is close and joint is effectively closed
     * @return false when no object is near the gripper. joint is not closed.
     */
    bool attemptClose(float additional_offset = 0)
    {
        DcHandle rb_0 = mDc->getRigidBody(mProps.parentPath.c_str());
        if (!rb_0)
        {
            CARB_LOG_ERROR("Parent rigid Body handle not valid for prim %s", mProps.parentPath.c_str());
            return false;
        }
        DcTransform t_0 = mDc->getRigidBodyPose(rb_0);
        DcTransform* offsetPtr = reinterpret_cast<DcTransform*>(&mProps.offset);
        DcTransform _adjustedTransform = (t_0 * (*offsetPtr));
        carb::Float3 dir = getBasisVectorX(_adjustedTransform.r);
        // CARB_LOG_INFO("gripper offset: (%f, %f, %f)", mProps.offset.p.x, mProps.offset.p.y, mProps.offset.p.z);
        // CARB_LOG_INFO(
        //     "gripper position: (%f, %f, %f)", _adjustedTransform.p.x, _adjustedTransform.p.y,
        //     _adjustedTransform.p.z);
        // CARB_LOG_INFO("gripper direction: (%f, %f, %f)", dir.x, dir.y, dir.z);
        // DcTransform threshOffset;
        // threshOffset.p.x = mProps.gripThreshold;
        // _t_0 = _t_0 * threshOffset; //Disabling until we get soft meshes for grippers
        uint32_t attempts = 1000000;
        additional_offset = 0.0f;
        omni::physx::RaycastHit result;
        bool hit = false;
        while (attempts)
        {
            attempts--;
            carb::Float3 p = isaacsim::core::utils::math::operator+(_adjustedTransform.p, dir* additional_offset);
            hit = mPhysxQuery->raycastClosest(p, dir, mProps.gripThreshold, result, false);

            if (hit)
            {

                DcHandle body = mDc->getRigidBody(intToPath(result.rigidBody).GetString().c_str());
                if (body == rb_0)
                {
                    additional_offset += 1.0e-3f;
                    continue;
                }
                CARB_LOG_INFO("Gripping prim %s at distance %f with parent %s",
                              intToPath(result.rigidBody).GetString().c_str(), result.distance,
                              mProps.parentPath.c_str());
                DcTransform t_1 = inverse(mDc->getRigidBodyPose(body)) * _adjustedTransform;

                mJointProperties.body0 = rb_0;
                mJointProperties.body1 = body;
                mDc->setRigidBodyDisableGravity(mJointProperties.body1, mProps.disableGravity);
                DcTransform* offsetPtr = reinterpret_cast<DcTransform*>(&mProps.offset);
                mJointProperties.pose0 = *offsetPtr;
                mJointProperties.pose1 = t_1;
                mJointProperties.axes = kDcAxisAll;


                mJointProperties.stiffness = mProps.stiffness;
                mJointProperties.damping = mProps.damping;
                mJointProperties.limitStiffness = mProps.stiffness;
                mJointProperties.limitDamping = mProps.damping;
                if (mProps.bendAngle > 0)
                {
                    mJointProperties.softLimit = true;
                    mJointProperties.lowerLimit = mProps.bendAngle;
                    mJointProperties.upperLimit = mProps.bendAngle;
                    mJointProperties.jointType = DcJointType::eSpherical;
                    mJointProperties.hasLimits[4] = true;
                    mJointProperties.hasLimits[5] = true;
                }
                else
                {
                    mJointProperties.hasLimits[4] = false;
                    mJointProperties.hasLimits[5] = false;
                    mJointProperties.softLimit = false;
                    mJointProperties.lowerLimit = 0;
                    mJointProperties.upperLimit = 0;
                    mJointProperties.jointType = DcJointType::eFixed;
                }


                mJointProperties.forceLimit = mProps.forceLimit;
                mJointProperties.torqueLimit = mProps.torqueLimit;
                if (!mJointHandle)
                {
                    std::string s(mProps.d6JointPath);
                    mJointProperties.name = (char*)(s).c_str();
                    mJointHandle = mDc->createD6Joint(&mJointProperties);
                }
                mDc->setD6JointProperties(mJointHandle, &mJointProperties);

                mIsClosed = true;
            }
            break;
        }
        if (attempts == 0)
        {
            CARB_LOG_INFO("Raycast Failed");
        }
        if (additional_offset)
        {
            CARB_LOG_WARN(
                "Surface Gripper is inside the parent Rigid body collider. please move it forward in the X offset direction by %f to avoid wasted computation",
                additional_offset);
        }
        return hit;
    }
    /**
     * @brief opens the Surface Gripper, releasing the object
     *
     * @return true
     */
    bool open()
    {
        if (!mIsInitialized)
        {
            CARB_LOG_ERROR("Please call initialize before opening");
            return false;
        }
        mAttemptClose = false;
        if (mIsClosed)
        {
            if (mDc->isSimulating())
            {
                mDc->wakeUpRigidBody(mJointProperties.body1);
                mDc->setRigidBodyDisableGravity(mJointProperties.body1, false);
            }
            mDc->destroyD6Joint(mJointHandle);
            mJointHandle = omni::isaac::dynamic_control::kDcInvalidHandle;

            mIsClosed = false;
            return true;
        }
        return false;
    }

private:
    DynamicControl* mDc = nullptr;
    omni::physx::IPhysxSceneQuery* mPhysxQuery = nullptr;
    DcHandle mJointHandle = omni::isaac::dynamic_control::kDcInvalidHandle;

    DcD6JointProperties mJointProperties;
    SurfaceGripperProperties mProps;

    bool mIsClosed;
    bool mIsInitialized;
    bool mAttemptClose;
};

} // omni
} // isaac
} // utils
