#!/usr/bin/python3

# Create a wrapper around the official WPILib ntproperty function
# This is needed so that the Finder classes can fake using ntproperties when NetworkTables is not installed.
#  Much easier development/debugging that way.

__all__ = ["ntproperty", ]

try:
    from networktables import NetworkTablesInstance
    NetworkTables = NetworkTablesInstance.getDefault()

    class _NtProperty:
        def __init__(self, key: str, defaultValue, writeDefault: bool, persistent: bool, inst: NetworkTableIsnstance) -> None:
            self.inst = inst

            self.ntvalue = self.inst.getGlobalAutoUpdateValue(key, defaultValue, writeDefault)
            self.ntvalue = self.inst.getEntry(key)
            # never overwrite persistent values with defaults
            if persistent:
                self.ntvalue.setPersistent()
            elif writeDefault:
                self.ntvalue.setDefaultValue(defaultValue)

            # this is an optimization, but presumes the value type never changes
            self.mkv = Value.getFactoryByType(self.ntvalue.getType())

            if hasattr(self.inst, "_api"):
                self.set = self._set_pynetworktables
            else:
                self.set = self._set_pyntcore
            
        def get(self, _):
            return self.ntvalue.value

        def _set_pynetworktables(self, _, value):
            self.inst._api.setEntryValueById(self.ntvalue._local_id, self.mkv(value))

        def _set_pyntcore(self, _, value):
            self.ntvalue.setValue(self.mkv(value))


    def ntproperty(key: str, defaultValue, writeDefault: bool = True, doc: str = None,
                   persistent: bool = False, *, inst: NetworkTableInstance = NetworkTables) -> property:
        """
        Replacement for WPILib ntproperty to be used when NT is not loaded on the machine

        :param key: A full NetworkTables key (eg ``/SmartDashboard/foo``) - ignored
        :param defaultValue: Default value to use if not in the table
        :type  defaultValue: any
        :param writeDefault: If True, put the default value to the table,
                             overwriting existing values
        :param doc: If given, will be the docstring of the property.
        :param persistent: If True, persist set values across restarts.
                           *writeDefault* is ignored if this is True.
        """

        ntprop = _NtProperty(key, defaultValue, writeDefault, persistent, inst)
        try:
            NetworkTables._ntproperties.add(ntprop)
        except AttributeError:
            pass  # pyntcore compat

        return property(fget=ntprop.get, fset=ntprop.set, doc=doc)

except Exception:
    # print('No WPILib ntproperty')

    class _NtProperty:
        def __init__(self, key: str, defaultValue, writeDefault: bool, persistent: bool) -> None:
            self._value = None
            self.set(None, defaultValue)
            return

        def get(self, _):
            return self._value

        def set(self, _, value):
            # NT does not have int storage, so convert to float to get the same behavior
            if isinstance(value, int):
                self._value = float(value)
            else:
                self._value = value
            return


    def ntproperty(key: str, defaultValue, writeDefault: bool = True, doc: str = None,
                   persistent: bool = False, *, inst=None) -> property:
        """
        Replacement for WPILib ntproperty to be used when NT is not loaded on the machine

        :param key: A full NetworkTables key (eg ``/SmartDashboard/foo``) - ignored
        :param defaultValue: Default value to use if not in the table
        :type  defaultValue: any
        :param writeDefault: If True, put the default value to the table,
                             overwriting existing values
        :param doc: If given, will be the docstring of the property.
        :param persistent: If True, persist set values across restarts.
                           *writeDefault* is ignored if this is True.
        """

        ntprop = _NtProperty(key, defaultValue, writeDefault, persistent)
        return property(fget=ntprop.get, fset=ntprop.set, doc=doc)


if __name__ == '__main__':
    # test cases

    class TestClass:
        test_val = ntproperty('/SmartDashboard/vision/test_value', 30, doc='Test value')

    test_inst = TestClass()
    print('test_val initial value', test_inst.test_val)
    for i in range(3):
        test_inst.test_val = i
        print('set test_val', i, test_inst.test_val)
