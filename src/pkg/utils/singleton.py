
##
# @class    Singleton
# @brief    Template to make a singleton class.
# @remark   Inherit this class to make a class a singleton.
#           Do not call the class constructor directly, but call <class name>.instance() to get singleton instance.
class Singleton:
    __instance = None

    @classmethod
    def __getInstance(cls):
        return cls.__instance

    @classmethod
    def instance(cls, *args, **kargs):
        cls.__instance = cls(*args, **kargs)
        cls.instance = cls.__getInstance
        return cls.__instance