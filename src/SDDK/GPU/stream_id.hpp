#ifndef __STREAM_ID_HPP__
#define __STREAM_ID_HPP__

/// Helper class to wrap stream id (integer number).
class stream_id
{
  private:
    int id_;
  public:
    explicit stream_id(int id__)
        : id_(id__)
    {
    }
    inline int id() const
    {
        return id_;
    }
};

#endif
