#!/usr/bin/python3

# Create a wrapper around the official WPILib ntproperty function
# This is needed so that the Finder classes can fake using ntproperties when NetworkTables is not installed.
#  Much easier development/debugging that way.

__all__ = ["ntproperty", ]

try:
    raise Exception
    from networktables.util import ntproperty
except Exception:
    print('No WPILib ntproperty')

    class _NtProperty:
        def __init__(self, defaultValue) -> None:
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

    def ntproperty(key: str, defaultValue, *, doc: str = None) -> property:
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

        ntprop = _NtProperty(defaultValue)
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
